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
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    return green_mask


def _filter_red(img):
    """HSV 筛选红色"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    return red_mask


def _calc_angle_top(img):
    """顶部视角：轮廓面积比算角度"""
    green_mask = _filter_green(img)
    red_mask = _filter_red(img)

    green_cnts, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not green_cnts and not red_cnts:
        return None

    if not green_cnts:
        return 0.0
    if not red_cnts:
        return 80.0

    green_cnts = sorted(green_cnts, key=cv2.contourArea, reverse=True)[:2]
    red_cnts = sorted(red_cnts, key=cv2.contourArea, reverse=True)[:2]

    green_area = sum(cv2.contourArea(c) for c in green_cnts)
    total_area = green_area + sum(cv2.contourArea(c) for c in red_cnts)

    if total_area == 0:
        return None

    angle = (green_area / total_area) * 80.0
    return round(min(max(angle, 0.0), 80.0), 1)


def _calc_angle_side(img):
    """侧面视角：轮廓宽度比算角度"""
    green_mask = _filter_green(img)
    red_mask = _filter_red(img)

    green_cnts, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not green_cnts and not red_cnts:
        return None

    if not green_cnts:
        return 0.0
    if not red_cnts:
        return 80.0

    green_cnts = sorted(green_cnts, key=cv2.contourArea, reverse=True)[:2]
    red_cnts = sorted(red_cnts, key=cv2.contourArea, reverse=True)[:2]

    def total_width(cnts):
        all_points = np.vstack(cnts)
        x_min = all_points[:, :, 0].min()
        x_max = all_points[:, :, 0].max()
        return x_max - x_min

    green_w = total_width(green_cnts)
    red_w = total_width(red_cnts)
    total_w = green_w + red_w

    if total_w == 0:
        return None

    angle = (green_w / total_w) * 80.0
    return round(min(max(angle, 0.0), 80.0), 1)


def predict_cv(image_path: str, view: str = "top") -> float:
    """
    使用 OpenCV 预测阀门开度角度

    Args:
        image_path: 图片路径
        view: 视角 "top" 或 "side"

    Returns:
        预测角度 (0-80)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    if view == "side":
        angle = _calc_angle_side(img)
    else:
        angle = _calc_angle_top(img)

    if angle is not None:
        return angle

    # 兜底：HoughCircles + 像素比
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=20, maxRadius=200)

    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    if circles is not None:
        circles = np.uint16(np.around(circles))
        c = max(circles[0], key=lambda c: c[2])
        cv2.circle(mask, (c[0], c[1]), c[2], 255, -1)

    green_mask = _filter_green(img)
    green_mask = cv2.bitwise_and(green_mask, mask)

    total_pixels = cv2.countNonZero(mask)
    green_pixels = cv2.countNonZero(green_mask)

    if total_pixels == 0:
        return 0.0

    angle = (green_pixels / total_pixels) * 80.0
    return round(min(max(angle, 0.0), 80.0), 1)
