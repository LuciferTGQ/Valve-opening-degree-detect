"""
对原始 top 图片做 CV 裁剪，提取阀门圆形区域
流程：遍历 origin data/top/ → 色块质心+percentile半径定位 → 裁剪 → 保存到 data_cropped_original/top/
裁剪失败则跳过，让用户手动检查
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np

from src.top.cvnew_predictor import (
    _filter_green, _filter_red, _find_valve_circle,
    _refine_center,
)


def _read_image(image_path: str):
    """兼容中文路径的图片读取"""
    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _find_valve_circle_relaxed(img):
    """多轮 HoughCircles，逐渐放宽参数"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape[:2]
    short = min(h, w)

    # 从严到宽多轮尝试
    attempts = [
        (1.2, 100, 30, max(10, int(short * 0.1)), int(short * 0.48)),
        (1.2, 80, 25, max(10, int(short * 0.08)), int(short * 0.55)),
        (1.5, 60, 20, max(8, int(short * 0.08)), int(short * 0.60)),
    ]

    best_circle = None
    for dp, p1, p2, min_r, max_r in attempts:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=short * 0.3,
                                   param1=p1, param2=p2, minRadius=min_r, maxRadius=max_r)
        if circles is not None and len(circles[0]) > 0:
            circles = np.uint16(np.around(circles))
            # 取最大的圆（最可能是阀门圆）
            c = max(circles[0], key=lambda c: c[2])
            if best_circle is None or c[2] > best_circle[2]:
                best_circle = (int(c[0]), int(c[1]), int(c[2]))

    return best_circle


def _estimate_radius_percentile(cnts, cx0, cy0, percentile=85):
    """用色块轮廓点到质心的距离 percentile 估算半径，排除散布噪声"""
    all_points = np.vstack([c.reshape(-1, 2) for c in cnts])
    distances = np.sqrt((all_points[:, 0] - cx0) ** 2 + (all_points[:, 1] - cy0) ** 2)
    radius = int(np.percentile(distances, percentile))
    return max(radius, 20)


def _locate_valve_for_crop(img):
    """定位阀门圆心+半径，用于裁剪。

    1. 全图红绿色块面积加权质心
    2. percentile 距离估算半径（排除散布噪声）
    3. 多轮 HoughCircles 尝试（给出更好的半径）
    4. 几何法精确定心
    返回 (cx, cy, radius) 或 None
    """
    h, w = img.shape[:2]

    green_mask = _filter_green(img)
    red_mask = _filter_red(img)
    combined = cv2.bitwise_or(green_mask, red_mask)

    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 100]

    if not cnts:
        return None

    # 面积加权质心
    total_area = 0
    sum_x = 0
    sum_y = 0
    for c in cnts:
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        sum_x += (M['m10'] / M['m00']) * area
        sum_y += (M['m01'] / M['m00']) * area
        total_area += area

    if total_area == 0:
        return None

    cx0 = int(sum_x / total_area)
    cy0 = int(sum_y / total_area)

    # percentile 半径估算（比 bounding box 更紧，排除散布噪声）
    radius0 = _estimate_radius_percentile(cnts, cx0, cy0, percentile=85)
    # 保底：至少 1/5 的短边
    if radius0 < min(h, w) // 5:
        radius0 = min(h, w) // 5

    # 多轮 HoughCircles（可能给出更好的半径）
    circle = _find_valve_circle_relaxed(img)
    if circle is not None:
        hcx, hcy, hr = circle
        dist = np.sqrt((hcx - cx0) ** 2 + (hcy - cy0) ** 2)
        # 如果 Hough 圆心离质心不远，采纳 Hough 的半径
        if dist < radius0 * 2.5:
            radius0 = hr

    # 几何法精确定心
    cx, cy = _refine_center(img, cx0, cy0, radius0)

    return (cx, cy, radius0)


def crop_single_top_image(img_path: str) -> np.ndarray | None:
    """对单张 top 图片做 CV 裁剪，返回裁剪后的 BGR 图或 None"""
    img = _read_image(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    img_area = h * w

    result = _locate_valve_for_crop(img)
    if result is None:
        return None

    cx, cy, radius = result

    # 裁剪：圆心为中心，1.3 倍半径（percentile 半径更紧，1.3x 已足够覆盖）
    crop_r = int(radius * 1.3)

    x1 = max(0, cx - crop_r)
    y1 = max(0, cy - crop_r)
    x2 = min(w, cx + crop_r)
    y2 = min(h, cy + crop_r)

    cropped = img[y1:y2, x1:x2]

    if cropped.size == 0:
        return None

    crop_area = cropped.shape[0] * cropped.shape[1]
    # 跳过：裁剪几乎等于原图（无裁剪价值）或太小
    if crop_area >= img_area * 0.98 or crop_area < 1000:
        return None

    return cropped


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CV 裁剪原始 top 图片')
    parser.add_argument('--input', default='origin data/top', help='原始数据目录')
    parser.add_argument('--output', default='data_cropped_original/top', help='裁剪输出目录')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_dir = os.path.join(project_root, args.input)
    output_dir = os.path.join(project_root, args.output)

    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"输入: {len(files)} 张原始图片")
    print(f"输出: {args.output}\n")

    success = 0
    skip = 0

    for i, f in enumerate(files):
        img_path = os.path.join(input_dir, f)
        cropped = crop_single_top_image(img_path)

        if cropped is None:
            skip += 1
            continue

        out_path = os.path.join(output_dir, f)
        cv2.imencode('.jpg', cropped)[1].tofile(out_path)
        success += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}] 成功 {success}, 跳过 {skip}")

    print(f"\n裁剪完成！成功 {success} 张，跳过 {skip} 张")
    print(f"输出目录: {output_dir}")
    print(f"\n请检查裁剪效果，删除裁剪不好的图片后告诉我")


if __name__ == '__main__':
    main()
