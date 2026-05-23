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
    """HSV 筛选红色"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)


def _find_valve_circle(img):
    """HoughCircles 定位阀门圆"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=20, maxRadius=200)
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


def _refine_center_by_lines(img, cx0, cy0, radius0, mask_ratio=0.8):
    """
    几何法精确定位圆心：找扇形两条直边（半径）的延长线交点

    1. 用初始圆的 0.8 蒙版过滤色块
    2. 对色块轮廓用 HoughLinesP 检测直线
    3. 筛选出扇形的两条半径边（过滤掉弧线和噪声线）
    4. 两条半径线延长求交点 = 圆心
    """
    mask_r = int(radius0 * mask_ratio)
    h, w = img.shape[:2]
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx0, cy0), mask_r, 255, -1)

    green_mask = cv2.bitwise_and(_filter_green(img), circle_mask)
    red_mask = cv2.bitwise_and(_filter_red(img), circle_mask)

    # 合并红绿掩膜，找所有色块轮廓
    combined = cv2.bitwise_or(green_mask, red_mask)
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([c for c in cnts if cv2.contourArea(c) > 100], key=cv2.contourArea, reverse=True)[:4]

    if not cnts:
        return cx0, cy0

    # 在轮廓图上检测直线
    contour_img = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_img, cnts, -1, 255, 2)

    lines = cv2.HoughLinesP(contour_img, 1, np.pi / 180, threshold=20,
                            minLineLength=mask_r // 4, maxLineGap=mask_r // 4)
    if lines is None:
        return cx0, cy0

    # 筛选：只保留通过蒙版圆附近的线段（靠近圆心的线才是半径边）
    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 线段中点
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # 中点到初始圆心的距离
        dist_center = np.sqrt((mx - cx0) ** 2 + (my - cy0) ** 2)
        # 线段长度
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # 线段方向（归一化）
        dx, dy = x2 - x1, y2 - y1
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm < 1:
            continue
        dx, dy = dx / norm, dy / norm

        # 半径边的特征：方向大致指向圆心
        # 从线段中点指向圆心的方向
        to_center_x = cx0 - mx
        to_center_y = cy0 - my
        to_center_norm = np.sqrt(to_center_x ** 2 + to_center_y ** 2)
        if to_center_norm < 1:
            continue
        to_center_x /= to_center_norm
        to_center_y /= to_center_norm

        # 方向与指向圆心的方向的夹角余弦（取绝对值，因为方向可能反向）
        dot = abs(dx * to_center_x + dy * to_center_y)

        # 保留：方向大致指向圆心 且 长度合理的线
        if dot > 0.7 and length > mask_r * 0.15:
            candidates.append((x1, y1, x2, y2, length, dot))

    if len(candidates) < 2:
        return cx0, cy0

    # 按 length * dot 排序，取前几条
    candidates.sort(key=lambda c: c[4] * c[5], reverse=True)
    top_lines = candidates[:6]

    # 遍历线对，找交点
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
            # 交点应在蒙版圆内部
            dist = np.sqrt((ix - cx0) ** 2 + (iy - cy0) ** 2)
            if dist < mask_r * 0.6 and 0 <= ix < w and 0 <= iy < h:
                intersections.append((ix, iy))

    if not intersections:
        return cx0, cy0

    # 取中位数
    xs = [p[0] for p in intersections]
    ys = [p[1] for p in intersections]
    cx = int(np.median(xs))
    cy = int(np.median(ys))

    return cx, cy


def _calc_angle(img, cx, cy, radius, mask_ratio):
    """在指定圆心和蒙版比例下计算红绿轮廓面积比"""
    mask_r = int(radius * mask_ratio)
    h, w = img.shape[:2]
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), mask_r, 255, -1)

    green_mask = cv2.bitwise_and(_filter_green(img), circle_mask)
    red_mask = cv2.bitwise_and(_filter_red(img), circle_mask)

    green_cnts, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_cnts = sorted([c for c in green_cnts if cv2.contourArea(c) > 10], key=cv2.contourArea, reverse=True)[:2]
    red_cnts = sorted([c for c in red_cnts if cv2.contourArea(c) > 10], key=cv2.contourArea, reverse=True)[:2]

    if not green_cnts and not red_cnts:
        return 0.0
    if not green_cnts:
        return 0.0
    if not red_cnts:
        return 80.0

    green_area = sum(cv2.contourArea(c) for c in green_cnts)
    red_area = sum(cv2.contourArea(c) for c in red_cnts)
    total = green_area + red_area

    if total == 0:
        return 0.0

    angle = (green_area / total) * 80.0
    return round(min(max(angle, 0.0), 80.0), 1)


def predict_cvnew(image_path: str, refine_ratio: float = 0.8, calc_ratio: float = 0.7) -> float:
    """
    双重蒙版法预测阀门开度角度

    1. HoughCircles 初步定位圆
    2. refine_ratio(0.8) 蒙版过滤色块
    3. 对色块轮廓检测直线，找扇形两条半径边
    4. 半径边延长线交点 = 精确圆心
    5. 用新圆心 + calc_ratio(0.7) 蒙版计算面积比
    6. 红绿轮廓面积比 → 角度

    Args:
        image_path: 图片路径
        refine_ratio: 定位用蒙版比例（较大，保留更多色块用于定位）
        calc_ratio: 计算用蒙版比例（较小，避开边框红色噪声）

    Returns:
        预测角度 (0-80)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 1. 初步定位
    result = _find_valve_circle(img)
    if result is None:
        return _fallback_predict(img)

    cx0, cy0, radius0 = result

    # 2. 几何法精确定位圆心（用较大蒙版保留更多色块信息）
    cx, cy = _refine_center_by_lines(img, cx0, cy0, radius0, refine_ratio)

    # 3. 用新圆心 + 较小蒙版计算（避开边框红色噪声）
    return _calc_angle(img, cx, cy, radius0, calc_ratio)


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
