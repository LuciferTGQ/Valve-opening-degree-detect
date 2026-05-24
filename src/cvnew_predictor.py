import cv2
import numpy as np


def _filter_green(img):
    """HSV + RGB 双重筛选绿色"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    b, g, r = cv2.split(img)
    rgb_mask = ((g > r) & (g > b) & (g > 50)).astype(np.uint8) * 255
    green_mask = cv2.bitwise_and(hsv_mask, rgb_mask)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)


def _filter_red(img):
    """HSV + RGB 双重筛选红色"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)
    b, g, r = cv2.split(img)
    rgb_mask = ((r > g) & (r > b) & (r > 80)).astype(np.uint8) * 255
    red_mask = cv2.bitwise_and(red_mask, rgb_mask)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)


def _find_valve_circle(img):
    """HoughCircles 定位阀门圆（半径自适应图像大小）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape[:2]
    short = min(h, w)
    min_r = max(10, int(short * 0.1))
    max_r = int(short * 0.48)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=min_r, maxRadius=max_r)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        c = max(circles[0], key=lambda c: c[2])
        return int(c[0]), int(c[1]), int(c[2])
    return None


def _line_intersection(p1, d1, p2, d2):
    """求两条直线的交点，p=线上一点，d=方向向量"""
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-6:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t = (dx * d2[1] - dy * d2[0]) / cross
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def _color_centroid(img, cx0, cy0, radius0):
    """在 HoughCircles 完整圆内，计算红绿色块的面积加权质心"""
    h, w = img.shape[:2]
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx0, cy0), radius0, 255, -1)

    green_mask = cv2.bitwise_and(_filter_green(img), circle_mask)
    red_mask = cv2.bitwise_and(_filter_red(img), circle_mask)
    combined = cv2.bitwise_or(green_mask, red_mask)

    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 100]
    if not cnts:
        return cx0, cy0

    total_area = 0
    sum_x = 0
    sum_y = 0
    for c in cnts:
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx_c = M['m10'] / M['m00']
        cy_c = M['m01'] / M['m00']
        sum_x += cx_c * area
        sum_y += cy_c * area
        total_area += area

    if total_area == 0:
        return cx0, cy0

    return int(sum_x / total_area), int(sum_y / total_area)


def _refine_center(img, cx0, cy0, radius0, mask_ratio=0.8):
    """
    色块质心 + 几何法精确定位圆心

    1. 在 HoughCircles 完整圆内算红绿色块的面积加权质心
    2. 以质心为中心画蒙版，过滤色块
    3. 对色块轮廓检测直线，找扇形半径边
    4. 半径边延长线交点 = 精确圆心
    """
    cx_color, cy_color = _color_centroid(img, cx0, cy0, radius0)

    mask_r = int(radius0 * mask_ratio)
    h, w = img.shape[:2]
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx_color, cy_color), mask_r, 255, -1)

    green_mask = cv2.bitwise_and(_filter_green(img), circle_mask)
    red_mask = cv2.bitwise_and(_filter_red(img), circle_mask)

    combined = cv2.bitwise_or(green_mask, red_mask)
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([c for c in cnts if cv2.contourArea(c) > 100], key=cv2.contourArea, reverse=True)[:4]

    if not cnts:
        return cx_color, cy_color

    contour_img = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_img, cnts, -1, 255, 2)

    lines = cv2.HoughLinesP(contour_img, 1, np.pi / 180, threshold=20,
                            minLineLength=mask_r // 4, maxLineGap=mask_r // 4)
    if lines is None:
        return cx_color, cy_color

    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        dx, dy = x2 - x1, y2 - y1
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm < 1:
            continue
        dx, dy = dx / norm, dy / norm

        to_center_x = cx_color - mx
        to_center_y = cy_color - my
        to_center_norm = np.sqrt(to_center_x ** 2 + to_center_y ** 2)
        if to_center_norm < 1:
            continue
        to_center_x /= to_center_norm
        to_center_y /= to_center_norm

        dot = abs(dx * to_center_x + dy * to_center_y)

        if dot > 0.7 and length > mask_r * 0.15:
            candidates.append((x1, y1, x2, y2, length, dot))

    if len(candidates) < 2:
        return cx_color, cy_color

    candidates.sort(key=lambda c: c[4] * c[5], reverse=True)
    top_lines = candidates[:6]

    intersections = []
    for i in range(len(top_lines)):
        for j in range(i + 1, len(top_lines)):
            x1, y1, x2, y2 = top_lines[i][:4]
            p1 = (x1, y1)
            d1 = (x2 - x1, y2 - y1)

            x1, y1, x2, y2 = top_lines[j][:4]
            p2 = (x1, y1)
            d2 = (x2 - x1, y2 - y1)

            pt = _line_intersection(p1, d1, p2, d2)
            if pt is None:
                continue

            ix, iy = pt
            dist = np.sqrt((ix - cx_color) ** 2 + (iy - cy_color) ** 2)
            if dist < mask_r * 0.6 and 0 <= ix < w and 0 <= iy < h:
                intersections.append((ix, iy))

    if not intersections:
        return cx_color, cy_color

    xs = [p[0] for p in intersections]
    ys = [p[1] for p in intersections]
    cx = int(np.median(xs))
    cy = int(np.median(ys))

    return cx, cy


def _calc_angle_arc(img, cx, cy, radius, inner_ratio=0.25, outer_ratio=0.75, n_radii=20):
    """
    弧长比法计算角度：用多个同心圆截红绿扇形，根据圆周上的弧长比推算角度

    原理：红绿两个对称扇形组成圆环，任意半径的圆周被红绿各切出一段弧，
    弧长比 = 扇形角度比 = 开度比例。多组半径取平均，抗检测缺失。

    Args:
        inner_ratio: 采样起始半径比例（避开中心）
        outer_ratio: 采样结束半径比例（避开边缘）
        n_radii: 采样半径数
    """
    h, w = img.shape[:2]

    # 预算整张图的红绿掩膜
    green_mask_full = _filter_green(img)
    red_mask_full = _filter_red(img)

    r_inner = int(radius * inner_ratio)
    r_outer = int(radius * outer_ratio)
    if r_outer <= r_inner:
        r_outer = r_inner + 1

    ratios = []
    all_green = 0
    all_red = 0
    for r in np.linspace(r_inner, r_outer, n_radii):
        r = int(r)
        n_samples = max(360, int(2 * np.pi * r))
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        xs = (cx + r * np.cos(angles)).astype(int)
        ys = (cy + r * np.sin(angles)).astype(int)

        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs_v = xs[valid]
        ys_v = ys[valid]

        if len(xs_v) < 10:
            continue

        green_count = np.sum(green_mask_full[ys_v, xs_v] > 0)
        red_count = np.sum(red_mask_full[ys_v, xs_v] > 0)
        all_green += green_count
        all_red += red_count

        total = green_count + red_count
        if total > 20 and green_count > 5 and red_count > 5:
            ratios.append(green_count / total)

    # 有足够双色半径，取中位数
    if len(ratios) >= 3:
        angle = np.median(ratios) * 80.0
        return round(min(max(angle, 0.0), 80.0), 1)

    # 退化：用所有半径的总弧长比
    total = all_green + all_red
    if total > 0:
        angle = (all_green / total) * 80.0
        return round(min(max(angle, 0.0), 80.0), 1)

    return 0.0


def _fallback_predict(img):
    """HoughCircles 失败时的兜底"""
    green_mask = _filter_green(img)
    red_mask = _filter_red(img)

    green_cnts, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_cnts = sorted(green_cnts, key=cv2.contourArea, reverse=True)[:2]
    red_cnts = sorted(red_cnts, key=cv2.contourArea, reverse=True)[:2]

    green_area = sum(cv2.contourArea(c) for c in green_cnts) if green_cnts else 0
    red_area = sum(cv2.contourArea(c) for c in red_cnts) if red_cnts else 0
    total = green_area + red_area

    if total == 0:
        return 0.0

    angle = (green_area / total) * 80.0
    return round(min(max(angle, 0.0), 80.0), 1)


def predict_cvnew(image_path: str, refine_ratio: float = 0.8) -> float:
    """
    色块质心 + 几何法 + 弧长比法预测阀门开度角度

    1. HoughCircles 初步定位圆
    2. 在大圆内计算红绿色块质心，以质心为中心画蒙版
    3. 对蒙版内色块轮廓检测直线，找扇形半径边
    4. 半径边延长线交点 = 精确圆心
    5. 用多个同心圆截红绿扇形，根据弧长比推算角度

    Args:
        image_path: 图片路径
        refine_ratio: 定位用蒙版比例（较大，保留更多色块用于定位）

    Returns:
        预测角度 (0-80)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    result = _find_valve_circle(img)
    if result is None:
        return _fallback_predict(img)

    cx0, cy0, radius0 = result

    cx, cy = _refine_center(img, cx0, cy0, radius0, refine_ratio)

    return _calc_angle_arc(img, cx, cy, radius0)
