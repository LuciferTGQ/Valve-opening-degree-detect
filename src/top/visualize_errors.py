"""
可视化 CVnew 误差最大的样本处理流程

对每张图输出一个 2x3 拼图：
  [原图+质心圆]    [原图+几何圆]    [蒙版圆+色块]
  [Mask]           [Green raw]      [Red raw]

标注：ground truth / 预测值 / 误差
输出到 error_analysis/
"""

import os
import sys
import math

import cv2
import numpy as np

from src.top.cvnew_predictor import (
    _filter_green, _filter_red, _refine_center, predict_cvnew,
)

DATA_DIR = "origin data/top"
OUT_DIR = "error_analysis"
TOP_N = 10


def load_ground_truth(data_dir):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith((".jpg", ".png"))])
    gt = []
    for f in files:
        parts = f.split("_")
        angle = float(parts[1].replace(".jpg", "").replace(".png", ""))
        gt.append((f, angle))
    return gt


def predict_with_debug(image_path):
    """运行完整 CVnew 流程并返回中间结果"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    h, w = img.shape[:2]
    green_mask = _filter_green(img)
    red_mask = _filter_red(img)

    # Step 2: 色块质心
    combined = cv2.bitwise_or(green_mask, red_mask)
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 100]

    if not cnts:
        return {
            "img": img, "centroid": None, "center": None, "radius": None,
            "green_mask": green_mask, "red_mask": red_mask,
        }

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

    cx0 = int(sum_x / total_area)
    cy0 = int(sum_y / total_area)

    # 估算半径
    all_points = np.vstack([c.reshape(-1, 2) for c in cnts])
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    radius0 = int(max(x_max - x_min, y_max - y_min) / 2)
    if radius0 < 20:
        radius0 = min(h, w) // 4

    # Step 3: 几何法精确定位圆心
    cx, cy = _refine_center(img, cx0, cy0, radius0)

    # 蒙版圆内色块
    mask_r = int(radius0 * 0.8)
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), mask_r, 255, -1)
    green_in = cv2.bitwise_and(green_mask, circle_mask)
    red_in = cv2.bitwise_and(red_mask, circle_mask)

    green_area = cv2.countNonZero(green_in)
    red_area = cv2.countNonZero(red_in)
    total_color = green_area + red_area
    green_ratio = green_area / total_color if total_color > 0 else 0
    red_ratio = red_area / total_color if total_color > 0 else 0

    return {
        "img": img,
        "centroid": (cx0, cy0),
        "center": (cx, cy),
        "radius": radius0,
        "mask_r": mask_r,
        "green_mask": green_mask,
        "red_mask": red_mask,
        "green_in": green_in,
        "red_in": red_in,
        "green_area": green_area,
        "red_area": red_area,
        "green_ratio": green_ratio,
        "red_ratio": red_ratio,
    }


def make_panel(bgr_img, title, text_lines=None):
    """加标题栏和底部文字，返回 BGR 图像"""
    panel = bgr_img.copy()
    h, w = panel.shape[:2]

    bar_h = 32
    bar = np.full((bar_h, w, 3), (50, 50, 50), dtype=np.uint8)
    cv2.putText(bar, title, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    panel = np.vstack([bar, panel])

    if text_lines:
        lh = 22
        bot = np.full((lh * len(text_lines), w, 3), (30, 30, 30), dtype=np.uint8)
        for i, line in enumerate(text_lines):
            cv2.putText(bot, line, (6, 17 + i * lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        panel = np.vstack([panel, bot])

    return panel


def resize_to(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


def pad_to_width(img, target_w):
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    return np.hstack([img, np.full((h, target_w - w, 3), (20, 20, 20), dtype=np.uint8)])


def pad_to_height(img, target_h):
    h, w = img.shape[:2]
    if h >= target_h:
        return img
    return np.vstack([img, np.full((target_h - h, w, 3), (20, 20, 20), dtype=np.uint8)])


def draw_circle(img, cx, cy, r, color, thickness=2):
    """画圆和圆心"""
    vis = img.copy()
    cv2.circle(vis, (cx, cy), r, color, thickness)
    cv2.circle(vis, (cx, cy), 4, color, -1)
    return vis


def create_visualization(filename, gt_angle, pred_angle, debug, out_path):
    """为单张图片生成 2x3 拼图"""
    img = debug["img"]
    h, w = img.shape[:2]
    centroid = debug.get("centroid")
    center = debug.get("center")
    radius = debug.get("radius")
    mask_r = debug.get("mask_r")
    green_mask = debug["green_mask"]
    red_mask = debug["red_mask"]

    target_h = 300
    err = abs(gt_angle - pred_angle)

    # --- 1. 原图 + 色块质心圆 (黄色) ---
    if centroid and radius:
        img_centroid = draw_circle(img, centroid[0], centroid[1], radius, (0, 255, 255), 2)
        p1 = make_panel(resize_to(img_centroid, target_h), "Color Centroid",
                        [f"center=({centroid[0]},{centroid[1]}) r={radius}"])
    else:
        p1 = make_panel(resize_to(img, target_h), "Color Centroid", ["FAILED"])

    # --- 2. 原图 + 几何法精确圆心圆 (蓝色) ---
    if center and radius:
        img_refined = draw_circle(img, center[0], center[1], radius, (255, 0, 0), 2)
        p2 = make_panel(resize_to(img_refined, target_h), "Refined Center (Geometric)",
                        [f"center=({center[0]},{center[1]})",
                         f"GT={gt_angle:.1f} Pred={pred_angle:.1f} Err={err:.2f}"])
    else:
        p2 = make_panel(resize_to(img, target_h), "Refined Center", ["N/A"])

    # --- 3. 蒙版圆 (r*0.8) + 色块 ---
    if center and mask_r:
        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        green_in = debug.get("green_in")
        red_in = debug.get("red_in")
        if green_in is not None:
            mask_vis[:, :, 1] = green_in
        if red_in is not None:
            mask_vis[:, :, 2] = red_in
        # 画蒙版圆边界
        cv2.circle(mask_vis, (center[0], center[1]), mask_r, (100, 100, 100), 1)
        # 画色块质心
        if green_in is not None:
            green_cnts, _ = cv2.findContours(green_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            green_cnts = [c for c in green_cnts if cv2.contourArea(c) > 30]
            for c in green_cnts:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    gcx = int(M['m10'] / M['m00'])
                    gcy = int(M['m01'] / M['m00'])
                    cv2.circle(mask_vis, (gcx, gcy), 5, (0, 255, 0), -1)
        if red_in is not None:
            red_cnts, _ = cv2.findContours(red_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            red_cnts = [c for c in red_cnts if cv2.contourArea(c) > 30]
            for c in red_cnts:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    rcx = int(M['m10'] / M['m00'])
                    rcy = int(M['m01'] / M['m00'])
                    cv2.circle(mask_vis, (rcx, rcy), 5, (0, 0, 255), -1)
        p3 = make_panel(resize_to(mask_vis, target_h), f"Mask Circle (r*0.8={mask_r})",
                        [f"green blocks, red blocks"])
    else:
        p3 = make_panel(np.zeros((target_h, target_h, 3), dtype=np.uint8), "Mask Circle", ["N/A"])

    # --- 4. Mask (红绿叠加) ---
    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    green_in = debug.get("green_in")
    red_in = debug.get("red_in")
    if green_in is not None:
        mask_overlay[:, :, 1] = green_in
    if red_in is not None:
        mask_overlay[:, :, 2] = red_in

    green_area = debug.get("green_area", 0)
    red_area = debug.get("red_area", 0)
    green_ratio = debug.get("green_ratio", 0)
    red_ratio = debug.get("red_ratio", 0)

    p4 = make_panel(resize_to(mask_overlay, target_h), "Mask (green+red)",
                    [f"green={green_area}px ({green_ratio:.3f})",
                     f"red={red_area}px ({red_ratio:.3f})"])

    # --- 5. Green raw (全图检测) ---
    green_vis = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
    green_raw_area = cv2.countNonZero(green_mask)
    p5 = make_panel(resize_to(green_vis, target_h), "Green (raw)",
                    [f"area = {green_raw_area} px"])

    # --- 6. Red raw (全图检测) ---
    red_vis = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    red_raw_area = cv2.countNonZero(red_mask)
    p6 = make_panel(resize_to(red_vis, target_h), "Red (raw)",
                    [f"area = {red_raw_area} px"])

    # --- 拼接 2x3 ---
    panels = [p1, p2, p3, p4, p5, p6]
    max_w = max(p.shape[1] for p in panels)
    max_h = max(p.shape[0] for p in panels)
    panels = [pad_to_width(pad_to_height(p, max_h), max_w) for p in panels]

    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:6])
    grid = np.vstack([row1, row2])

    # --- 顶部标题栏 ---
    title_h = 44
    title_bar = np.full((title_h, grid.shape[1], 3), (40, 40, 80), dtype=np.uint8)
    title = f"{filename}  |  GT: {gt_angle:.1f}  Pred: {pred_angle:.1f}  Error: {err:.2f}"
    cv2.putText(title_bar, title, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    final = np.vstack([title_bar, grid])

    cv2.imwrite(out_path, final)


def main():
    data = load_ground_truth(DATA_DIR)
    paths = [os.path.join(DATA_DIR, d[0]) for d in data]

    print(f"加载 {len(data)} 张图片，计算误差...")
    preds = [predict_cvnew(p) for p in paths]
    errors = [(abs(gt - preds[i]), i) for i, (_, gt) in enumerate(data)]
    errors.sort(reverse=True)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n误差 Top {TOP_N}：")
    print(f"{'#':>3}  {'文件':<22}  {'GT':>5}  {'Pred':>5}  {'Err':>5}")
    print("-" * 50)

    for rank, (err, idx) in enumerate(errors[:TOP_N], 1):
        fname, gt = data[idx]
        pr = preds[idx]
        print(f"{rank:>3}  {fname:<22}  {gt:>5.1f}  {pr:>5.1f}  {err:>5.2f}")

        debug = predict_with_debug(os.path.join(DATA_DIR, fname))
        out_path = os.path.join(OUT_DIR, f"{rank:02d}_{fname.replace('.', '_')}.png")
        create_visualization(fname, gt, pr, debug, out_path)

    print(f"\n已输出 {min(TOP_N, len(errors))} 张到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
