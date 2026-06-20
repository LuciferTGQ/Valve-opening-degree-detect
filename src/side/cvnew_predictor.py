import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np


ANGLE_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _read_image(image_path: str) -> Optional[np.ndarray]:
    """兼容中文路径的图片读取方式，避免 Windows 下 cv2.imread 失败。"""
    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _enhance_if_dark(img: np.ndarray) -> np.ndarray:
    """画面偏暗时提升亮度，避免深绿色/深红色在阈值分割时漏检。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mean_v = float(np.mean(v))

    if mean_v >= 95:
        return img

    # 先把 V 通道按比例拉到较稳定的亮度，再用 CLAHE 提升局部对比度。
    scale = min(1.9, 125.0 / max(mean_v, 1.0))
    hsv[:, :, 2] = np.clip(v.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@dataclass
class SidePredictionDebug:
    """保存中间量，便于评估和误差可视化。"""

    raw_angle: float
    calibrated: bool
    geometry_angle: Optional[float]
    green_width: int
    red_width: int
    green_pixels: int
    red_pixels: int
    band_box: tuple[int, int, int, int]


def _filter_green(img: np.ndarray) -> np.ndarray:
    """用 HSV 阈值和 RGB 通道关系共同筛选绿色区域。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    b, g, r = cv2.split(img)
    # 只保留 G 通道明显占优的像素，降低背景和反光误检。
    rgb_mask = (
        (g > r)
        & (g > b)
        & (g > 50)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_and(hsv_mask, rgb_mask)
    # 开运算去掉零散噪点，保留主要色块。
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _filter_red(img: np.ndarray) -> np.ndarray:
    """用 HSV 阈值和 RGB 通道关系共同筛选红色区域。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 红色在 HSV 色相上跨 0 度，因此需要两个区间合并。
    mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
    hsv_mask = cv2.bitwise_or(mask1, mask2)
    b, g, r = cv2.split(img)
    # 只保留 R 通道明显占优的像素，避免把黄色/白色高光当成红色。
    rgb_mask = ((r > g) & (r > b) & (r > 80)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(hsv_mask, rgb_mask)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _complete_color_mask(mask: np.ndarray, min_area: Optional[float] = None) -> np.ndarray:
    """填充颜色轮廓，补回 OPEN/CLOSED 白字造成的空洞。"""
    h, w = mask.shape[:2]
    if min_area is None:
        min_area = max(80, h * w * 0.0004)

    # 先用闭运算连接被文字或反光轻微打断的色块。
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 只填充足够大的外轮廓；小噪点不参与补全。
    filled = np.zeros_like(mask)
    for cnt in cnts:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filled, [cnt], -1, 255, -1)

    return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _recover_nearby_green_fragments(green_raw: np.ndarray, green_mask: np.ndarray) -> np.ndarray:
    """把被反光切断、但仍靠近主绿色块的小绿色碎片补回。"""
    if int(np.count_nonzero(green_raw)) == 0 or int(np.count_nonzero(green_mask)) == 0:
        return green_mask

    raw_contours, _ = cv2.findContours(green_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_items = []
    for cnt in raw_contours:
        area = cv2.contourArea(cnt)
        if area >= 18:
            raw_items.append((area, cv2.boundingRect(cnt)))
    if len(raw_items) < 2:
        return green_mask

    # Only repair when a green fragment appears to the right of a larger green block
    # on roughly the same horizontal band. This avoids expanding already-good masks.
    should_recover = False
    for left_area, (lx, ly, lw, lh) in raw_items:
        if left_area < 300:
            continue
        left_right = lx + lw
        left_center_y = ly + lh / 2.0
        for right_area, (rx, ry, rw, rh) in raw_items:
            if right_area < 40 or rx <= left_right + 3:
                continue
            if rw > max(15, int(lw * 0.16)):
                continue
            gap = rx - left_right
            if gap > max(90, int(lw * 0.9)):
                continue
            y_overlap = max(0, min(ly + lh, ry + rh) - max(ly, ry))
            overlap_ratio = y_overlap / max(float(min(lh, rh)), 1.0)
            right_center_y = ry + rh / 2.0
            center_close = abs(right_center_y - left_center_y) <= max(lh, rh) * 0.65
            if overlap_ratio >= 0.25 or center_close:
                should_recover = True
                break
        if should_recover:
            break

    if not should_recover:
        return green_mask

    nearby = cv2.dilate(green_mask, np.ones((25, 61), np.uint8)) > 0

    recovered = green_mask.copy()
    for cnt in raw_contours:
        area = cv2.contourArea(cnt)
        if area < 18:
            continue

        component = np.zeros_like(green_mask)
        cv2.drawContours(component, [cnt], -1, 255, -1)
        already_kept = np.any((component > 0) & (green_mask > 0))
        near_main_green = np.any((component > 0) & nearby)
        if already_kept or near_main_green:
            cv2.drawContours(recovered, [cnt], -1, 255, -1)

    return cv2.morphologyEx(recovered, cv2.MORPH_CLOSE, np.ones((7, 25), np.uint8))


def _keep_red_adjacent_to_green(
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    max_gap: int = 12,
) -> np.ndarray:
    """只保留与绿色区域相邻的红色连通块，减少外部红色干扰。"""
    if int(np.count_nonzero(green_mask)) == 0 or int(np.count_nonzero(red_mask)) == 0:
        return red_mask

    kernel_size = max(3, max_gap * 2 + 1)
    green_nearby = cv2.dilate(green_mask, np.ones((kernel_size, kernel_size), np.uint8)) > 0
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kept = np.zeros_like(red_mask)
    for cnt in contours:
        component = np.zeros_like(red_mask)
        cv2.drawContours(component, [cnt], -1, 255, -1)
        if np.any((component > 0) & green_nearby):
            cv2.drawContours(kept, [cnt], -1, 255, -1)

    original_pixels = int(np.count_nonzero(red_mask))
    kept_pixels = int(np.count_nonzero(kept))
    # 如果邻接过滤删得太狠，说明绿色可能漏检或红绿之间有较大遮挡，退回原红色 mask。
    if kept_pixels < max(80, original_pixels * 0.18):
        return red_mask
    return cv2.morphologyEx(kept, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _recover_left_red_sliver(red_raw: np.ndarray, red_mask: np.ndarray, green_mask: np.ndarray) -> np.ndarray:
    """恢复高开度场景下紧贴主绿色块左侧、面积较小的红色窄条。"""
    if int(np.count_nonzero(red_raw)) == 0 or int(np.count_nonzero(green_mask)) == 0:
        return red_mask

    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not green_contours:
        return red_mask

    main_green = max(green_contours, key=cv2.contourArea)
    gx, gy, gw, gh = cv2.boundingRect(main_green)
    if gw <= 0 or gh <= 0:
        return red_mask

    red_contours, _ = cv2.findContours(red_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    recovered = red_mask.copy()
    changed = False

    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area < 250:
            continue

        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if rw <= 0 or rh <= 0:
            continue

        # 只恢复位于主绿色左侧、与主绿色高度重叠、且形态像窄红条的小连通块。
        y_overlap = max(0, min(gy + gh, ry + rh) - max(gy, ry))
        overlap_ratio = y_overlap / max(float(min(gh, rh)), 1.0)
        x_gap = gx - (rx + rw)
        left_of_green = rx < gx and x_gap <= max(35, int(gw * 0.18))
        narrow_sliver = rw <= max(45, int(gw * 0.22)) and rh >= max(18, int(gh * 0.25))
        not_bottom_reflection = ry < gy + gh * 0.75

        if left_of_green and overlap_ratio >= 0.35 and narrow_sliver and not_bottom_reflection:
            cv2.drawContours(recovered, [cnt], -1, 255, -1)
            changed = True

    if not changed:
        return red_mask
    return cv2.morphologyEx(recovered, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _recover_green_highlight_in_roi(
    img: np.ndarray,
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    band_box: tuple[int, int, int, int],
) -> np.ndarray:
    """只在已定位 ROI 内补回绿色高光区域，避免全图放宽阈值引入杂色。"""
    x, y, w, h = band_box
    if w <= 0 or h <= 0:
        return green_mask

    x1 = max(0, x)
    x2 = min(img.shape[1], x + w)
    y1 = max(0, y)
    y2 = min(img.shape[0], y + h)
    if y2 <= y1 or x2 <= x1:
        return green_mask

    img_roi = img[y1:y2, x1:x2]
    green_roi = green_mask[y1:y2, x1:x2]
    red_roi = red_mask[y1:y2, x1:x2]
    if int(np.count_nonzero(green_roi)) == 0:
        return green_mask

    hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 反光通常表现为高亮、低饱和；只补“贴着已有绿色”的部分，避免把金属高光整体染绿。
    highlight = (v > 145) & (s < 95)
    near_green = cv2.dilate(green_roi, np.ones((17, 17), np.uint8)) > 0
    near_red = cv2.dilate(red_roi, np.ones((7, 7), np.uint8)) > 0
    recover = (highlight & near_green & ~near_red).astype(np.uint8) * 255
    if int(np.count_nonzero(recover)) == 0:
        return green_mask

    repaired = green_mask.copy()
    repaired[y1:y2, x1:x2] = cv2.bitwise_or(green_roi, recover)
    return cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _recover_green_right_scan_in_roi(
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    band_box: tuple[int, int, int, int],
    green_hint_mask: Optional[np.ndarray] = None,
    allow_edge_only: bool = False,
) -> np.ndarray:
    """在 ROI 内从右向左寻找同高度绿色碎片，并补全主绿色块到碎片之间的缺口。"""
    x, y, w, h = band_box
    if w <= 0 or h <= 0:
        return green_mask

    x1 = max(0, x)
    x2 = min(green_mask.shape[1], x + w)
    y1 = max(0, y)
    y2 = min(green_mask.shape[0], y + h)
    if y2 <= y1 or x2 <= x1:
        return green_mask

    green_roi = green_mask[y1:y2, x1:x2]
    red_roi = red_mask[y1:y2, x1:x2]
    hint_source = green_hint_mask if green_hint_mask is not None else green_mask
    hint_roi = hint_source[y1:y2, x1:x2]
    if int(np.count_nonzero(green_roi)) == 0:
        return green_mask

    roi_h, roi_w = green_roi.shape[:2]
    combined_roi = cv2.bitwise_or(green_roi, red_roi)
    row_counts = np.sum(combined_roi > 0, axis=1)
    valid_rows = np.where(row_counts >= max(4, int(roi_w * 0.03)))[0]
    if valid_rows.size == 0:
        return green_mask

    repaired_roi = green_roi.copy()
    half_band = max(2, int(round(roi_h * 0.025)))
    max_gap = max(12, min(110, int(round(roi_w * 0.38))))
    repair_rows: set[int] = set()

    for row in valid_rows:
        band_top = max(0, row - half_band)
        band_bottom = min(roi_h, row + half_band + 1)
        hint_cols = np.where(np.any(hint_roi[band_top:band_bottom, :] > 0, axis=0))[0]
        if hint_cols.size < 2:
            continue

        hint_runs = _column_runs(hint_cols)
        if len(hint_runs) < 2:
            continue

        right_start, right_end = hint_runs[-1]
        _, left_end = hint_runs[-2]
        gap_width = right_start - left_end - 1
        right_width = right_end - right_start + 1
        if gap_width < 3 or gap_width > max_gap:
            continue
        if right_width > max(24, int(round(roi_w * 0.12))):
            continue

        red_gap = red_roi[band_top:band_bottom, left_end + 1 : right_start]
        gap_area = max(1, red_gap.size)
        if int(np.count_nonzero(red_gap)) > max(3, int(gap_area * 0.04)):
            continue

        row_start = max(0, row - half_band * 3)
        row_end = min(roi_h, row + half_band * 3 + 1)
        repair_rows.update(range(row_start, row_end))

    if allow_edge_only:
        repair_rows.update(int(row) for row in valid_rows)

    if not repair_rows:
        return green_mask

    for row in valid_rows:
        if int(row) not in repair_rows:
            continue

        band_top = max(0, row - half_band)
        band_bottom = min(roi_h, row + half_band + 1)

        # 用同一高度带的列投影判断绿色是否被高光切断，避免单行噪声影响判断。
        green_cols = np.where(np.any(green_roi[band_top:band_bottom, :] > 0, axis=0))[0]
        if green_cols.size < 2:
            continue

        green_runs = _column_runs(green_cols)
        if len(green_runs) < 2:
            green_runs = []

        row_green_cols = np.where(green_roi[row, :] > 0)[0]
        rightmost_green = int(row_green_cols[-1]) if row_green_cols.size else int(green_cols[-1])
        edge_gap_start = rightmost_green + 1
        edge_gap_width = roi_w - edge_gap_start
        if 3 <= edge_gap_width <= max_gap:
            red_edge_gap = red_roi[band_top:band_bottom, edge_gap_start:roi_w]
            gap_area = max(1, red_edge_gap.size)
            # 从 ROI 最右侧向左扫描遇到绿色后，若右侧空白没有红色，则补成绿色。
            if int(np.count_nonzero(red_edge_gap)) <= max(3, int(gap_area * 0.04)):
                repaired_roi[row, rightmost_green:roi_w] = 255

        if len(green_runs) < 2:
            continue

        # 从最右侧开始向左找：发现右侧绿色碎片后，只补它和左侧相邻绿色块之间的缺口。
        for idx in range(len(green_runs) - 1, 0, -1):
            right_start, _ = green_runs[idx]
            _, left_end = green_runs[idx - 1]
            gap_start = left_end + 1
            gap_end = right_start
            gap_width = gap_end - gap_start
            if gap_width < 3 or gap_width > max_gap:
                continue

            red_gap = red_roi[band_top:band_bottom, gap_start:gap_end]
            gap_area = max(1, red_gap.size)
            # 缺口里如果明显有红色，说明这里更可能是红绿边界或误检，不做绿色补全。
            if int(np.count_nonzero(red_gap)) > max(3, int(gap_area * 0.04)):
                continue

            repaired_roi[row, gap_start:gap_end] = 255
            break

    repaired = green_mask.copy()
    repaired[y1:y2, x1:x2] = repaired_roi
    return cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _recover_green_gap_after_red_in_roi(
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    band_box: tuple[int, int, int, int],
) -> np.ndarray:
    """在 ROI 内补全红色右侧到主绿色之间的反光缺口。"""
    x, y, w, h = band_box
    if w <= 0 or h <= 0:
        return green_mask

    x1 = max(0, x)
    x2 = min(green_mask.shape[1], x + w)
    y1 = max(0, y)
    y2 = min(green_mask.shape[0], y + h)
    if y2 <= y1 or x2 <= x1:
        return green_mask

    green_roi = green_mask[y1:y2, x1:x2]
    red_roi = red_mask[y1:y2, x1:x2]
    if int(np.count_nonzero(green_roi)) == 0 or int(np.count_nonzero(red_roi)) == 0:
        return green_mask

    roi_h, roi_w = green_roi.shape[:2]
    scan_y1 = max(0, int(roi_h * 0.20))
    scan_y2 = min(roi_h, int(roi_h * 0.85))
    if scan_y2 <= scan_y1:
        return green_mask

    repaired_roi = green_roi.copy()
    changed = False
    max_gap = max(8, min(90, int(round(roi_w * 0.32))))

    for row in range(scan_y1, scan_y2):
        red_runs = _column_runs(np.where(red_roi[row, :] > 0)[0])
        green_runs = _column_runs(np.where(green_roi[row, :] > 0)[0])
        if not red_runs or not green_runs:
            continue

        # 取最靠近绿色的红色右边界，以及其右侧第一个绿色段。
        red_right = max(end for _, end in red_runs)
        right_green_runs = [(start, end) for start, end in green_runs if start > red_right]
        if not right_green_runs:
            continue

        green_start, _ = min(right_green_runs, key=lambda run: run[0])
        gap_start = red_right + 1
        gap_end = green_start
        gap_width = gap_end - gap_start
        if gap_width < 3 or gap_width > max_gap:
            continue

        # 缺口里如果已有红色，不应染成绿色；这里只补纯反光/漏检造成的空白。
        if int(np.count_nonzero(red_roi[row, gap_start:gap_end])) > 0:
            continue

        repaired_roi[row, gap_start:gap_end] = 255
        changed = True

    if not changed:
        return green_mask

    repaired = green_mask.copy()
    repaired[y1:y2, x1:x2] = repaired_roi
    return cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))


def _fill_green_gap_after_red_crop(green_crop: np.ndarray, red_crop: np.ndarray) -> np.ndarray:
    """在测量坐标系中逐行补全反光造成的绿色缺口。"""
    if int(np.count_nonzero(green_crop)) == 0 or int(np.count_nonzero(red_crop)) == 0:
        return green_crop

    h, w = green_crop.shape[:2]
    repaired = green_crop.copy()
    max_gap = max(8, min(90, int(round(w * 0.32))))
    changed = False

    for row in range(h):
        red_runs = _column_runs(np.where(red_crop[row, :] > 0)[0])
        green_runs = _column_runs(np.where(green_crop[row, :] > 0)[0])
        if not green_runs:
            continue

        # 同一条带上绿色可能被反光切成多段；只要段间没有红色，就连接这些绿色段。
        if len(green_runs) >= 2:
            for (left_start, left_end), (right_start, right_end) in zip(green_runs, green_runs[1:]):
                internal_gap_start = left_end + 1
                internal_gap_end = right_start
                internal_gap_width = internal_gap_end - internal_gap_start
                if internal_gap_width < 3 or internal_gap_width > max_gap:
                    continue
                if int(np.count_nonzero(red_crop[row, internal_gap_start:internal_gap_end])) > 0:
                    continue
                repaired[row, internal_gap_start:internal_gap_end] = 255
                changed = True

        if not red_runs:
            continue

        red_right = max(end for _, end in red_runs)
        right_green_runs = [(start, end) for start, end in green_runs if start > red_right]
        if not right_green_runs:
            continue

        green_start, _ = min(right_green_runs, key=lambda run: run[0])
        gap_start = red_right + 1
        gap_end = green_start
        gap_width = gap_end - gap_start
        if gap_width < 3 or gap_width > max_gap:
            continue
        if int(np.count_nonzero(red_crop[row, gap_start:gap_end])) > 0:
            continue

        repaired[row, gap_start:gap_end] = 255
        changed = True

    if not changed:
        return green_crop
    return cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))


def _rough_measure_crop(green_crop: np.ndarray, red_crop: np.ndarray) -> tuple[int, int, Optional[float]]:
    """快速估计 ROI 测量坐标系内的红绿宽度和几何角，用于决定是否启用高开度补绿。"""
    roi_h, roi_w = green_crop.shape[:2]
    y1 = max(0, int(roi_h * 0.30))
    y2 = min(roi_h, int(roi_h * 0.70))
    if y2 <= y1 or roi_w <= 0:
        return 0, 0, None

    sample_count = min(31, max(7, y2 - y1))
    sample_rows = np.linspace(y1, y2 - 1, sample_count).astype(int)
    row_measurements = []
    for row in sample_rows:
        green_runs = _column_runs(np.where(green_crop[row, :] > 0)[0])
        red_runs = _column_runs(np.where(red_crop[row, :] > 0)[0])
        green_width = max((end - start + 1 for start, end in green_runs), default=0)
        red_width = max((end - start + 1 for start, end in red_runs), default=0)
        line_angle = _projection_calibration_from_widths(green_width, red_width) if green_width > 0 and red_width > 0 else None
        if line_angle is not None:
            row_measurements.append((green_width, red_width, green_width + red_width, line_angle))

    if not row_measurements:
        return 0, 0, None

    rows = np.array(row_measurements, dtype=np.float32)
    if len(rows) >= 5:
        low, high = np.percentile(rows[:, 2], [20, 80])
        filtered = rows[(rows[:, 2] >= low) & (rows[:, 2] <= high)]
        if len(filtered) >= 3:
            rows = filtered

    green_width = int(round(float(np.median(rows[:, 0]))))
    red_width = int(round(float(np.median(rows[:, 1]))))
    geometry_angle = float(np.median(rows[:, 3]))
    return green_width, red_width, geometry_angle


def _main_color_band_projection_legacy(
    green_mask: np.ndarray, red_mask: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """从补全后的红绿 mask 中定位侧面指示器的主颜色带。"""
    combined = cv2.bitwise_or(green_mask, red_mask)
    h, w = combined.shape[:2]

    # 用横向闭运算把被文字、刻度或轻微反光切开的同一条色带连接起来。
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((7, 21), np.uint8))

    row_counts = np.sum(combined > 0, axis=1)
    if row_counts.size == 0 or int(row_counts.max()) == 0:
        return np.zeros_like(combined), (0, 0, 0, 0)

    # 彩色指示带是横向颜色像素最密集的一段区域；按行投影比“最大连通块”
    # 更能避开侧面反射小色块和远处背景误检。
    row_threshold = max(5, int(row_counts.max() * 0.18))
    row_runs = _column_runs(np.where(row_counts >= row_threshold)[0])
    if not row_runs:
        return np.zeros_like(combined), (0, 0, 0, 0)

    candidates = []
    for y_start, y_end in row_runs:
        band = combined[y_start : y_end + 1, :]
        band_h = y_end - y_start + 1
        col_counts = np.sum(band > 0, axis=0)
        col_threshold = max(2, int(band_h * 0.08))
        col_runs = _column_runs(np.where(col_counts >= col_threshold)[0])
        if not col_runs:
            continue

        # 取该水平带内最宽的连续颜色列段作为主 ROI，排除左右边缘反射色块。
        x_start, x_end = max(col_runs, key=lambda run: run[1] - run[0])
        score = float(np.sum(row_counts[y_start : y_end + 1])) * (x_end - x_start + 1)
        candidates.append((score, x_start, y_start, x_end, y_end))

    if not candidates:
        return np.zeros_like(combined), (0, 0, 0, 0)

    _, x_start, y_start, x_end, y_end = max(candidates, key=lambda item: item[0])

    pad_x = max(2, int((x_end - x_start + 1) * 0.02))
    pad_y = max(2, int((y_end - y_start + 1) * 0.15))
    x = max(0, x_start - pad_x)
    y = max(0, y_start - pad_y)
    x2 = min(w, x_end + pad_x + 1)
    y2 = min(h, y_end + pad_y + 1)
    bw = x2 - x
    bh = y2 - y

    band_mask = np.zeros_like(combined)
    band_mask[y:y2, x:x2] = 255
    return band_mask, (x, y, bw, bh)


def _column_runs(cols: np.ndarray) -> list[tuple[int, int]]:
    """把一组列坐标合并成连续区间，用于计算单条扫描线上的色块宽度。"""
    if len(cols) == 0:
        return []
    cols = np.array(sorted(set(int(c) for c in cols)))
    runs = []
    start = prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col <= prev + 2:
            prev = col
        else:
            runs.append((start, prev))
            start = prev = col
    runs.append((start, prev))
    return runs


def _largest_run_width(mask: np.ndarray, x1: int, x2: int, threshold: int) -> int:
    """旧的按列投影宽度计算工具，保留给调试或后续对比使用。"""
    if x2 <= x1:
        return 0
    col_counts = np.sum(mask[:, x1:x2] > 0, axis=0)
    cols = np.where(col_counts > threshold)[0] + x1
    runs = _column_runs(cols)
    if not runs:
        return 0
    return max(end - start + 1 for start, end in runs)


def _row_band_runs(mask: np.ndarray, row: int, half_band: int) -> list[tuple[int, int]]:
    """在采样行附近取一个小高度带做列投影，降低镜头畸变/色带弯曲对单行测宽的影响。"""
    h = mask.shape[0]
    y1 = max(0, row - half_band)
    y2 = min(h, row + half_band + 1)
    cols = np.where(np.any(mask[y1:y2, :] > 0, axis=0))[0]
    return _column_runs(cols)


def _scan_angle_std(green_crop: np.ndarray, red_crop: np.ndarray) -> Optional[float]:
    """评估不旋转时的扫描线角度波动；若已经稳定，就没有必要做 deskew。"""
    roi_h, roi_w = green_crop.shape[:2]
    y1 = max(0, int(roi_h * 0.30))
    y2 = min(roi_h, int(roi_h * 0.70))
    if y2 <= y1 or roi_w <= 0:
        return None

    sample_count = min(31, max(7, y2 - y1))
    sample_rows = np.linspace(y1, y2 - 1, sample_count).astype(int)
    row_measurements = []
    for row in sample_rows:
        green_runs = _column_runs(np.where(green_crop[row, :] > 0)[0])
        red_runs = _column_runs(np.where(red_crop[row, :] > 0)[0])
        green_width = max((end - start + 1 for start, end in green_runs), default=0)
        red_width = max((end - start + 1 for start, end in red_runs), default=0)
        line_angle = _projection_calibration_from_widths(green_width, red_width) if green_width > 0 and red_width > 0 else None
        if line_angle is not None:
            row_measurements.append((green_width + red_width, line_angle))

    if len(row_measurements) < 5:
        return None
    rows = np.array(row_measurements, dtype=np.float32)
    low, high = np.percentile(rows[:, 0], [20, 80])
    filtered = rows[(rows[:, 0] >= low) & (rows[:, 0] <= high)]
    if len(filtered) >= 3:
        rows = filtered
    return float(np.std(rows[:, 1]))


def _estimate_band_boundary_angle(green_crop: np.ndarray, red_crop: np.ndarray) -> Optional[float]:
    """拟合颜色带上下边界的共同斜率，用于让扫描条带与上下边界平行。"""
    combined = cv2.bitwise_or(green_crop, red_crop)
    if int(np.count_nonzero(combined)) < 40:
        return None

    h, w = combined.shape[:2]
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    xs = []
    tops = []
    bottoms = []
    min_col_pixels = max(3, int(h * 0.03))
    for col in range(w):
        ys = np.where(closed[:, col] > 0)[0]
        if len(ys) >= min_col_pixels:
            xs.append(col)
            tops.append(int(ys[0]))
            bottoms.append(int(ys[-1]))

    if len(xs) < max(20, int(w * 0.15)):
        return None

    xs_arr = np.array(xs, dtype=np.float32)
    top_arr = np.array(tops, dtype=np.float32)
    bottom_arr = np.array(bottoms, dtype=np.float32)

    # 排除左右端帽、竖向拖尾和孤立噪声，对主体上下边界拟合更稳定。
    low, high = np.percentile(xs_arr, [10, 90])
    keep = (xs_arr >= low) & (xs_arr <= high)
    xs_arr = xs_arr[keep]
    top_arr = top_arr[keep]
    bottom_arr = bottom_arr[keep]
    if len(xs_arr) < 10:
        return None

    slopes = []
    for ys in (top_arr, bottom_arr, (top_arr + bottom_arr) / 2.0):
        slope, _ = np.polyfit(xs_arr, ys, 1)
        slopes.append(float(slope))

    return math.degrees(math.atan(float(np.median(slopes))))


def _rotate_mask_pair(
    green_crop: np.ndarray,
    red_crop: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按同一角度旋转红绿 mask，保持两者几何关系一致。"""
    h, w = green_crop.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    green_rotated = cv2.warpAffine(green_crop, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    red_rotated = cv2.warpAffine(red_crop, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return green_rotated, red_rotated


def _deskew_band_masks(
    green_crop: np.ndarray,
    red_crop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """根据色带主方向做小角度矫正，避免倾斜色带仍按水平扫描线测宽。"""
    combined = cv2.bitwise_or(green_crop, red_crop)
    points_yx = np.column_stack(np.where(combined > 0))
    if len(points_yx) < 40:
        return green_crop, red_crop

    points_xy = points_yx[:, ::-1].astype(np.float32)
    rect = cv2.minAreaRect(points_xy)
    rect_w, rect_h = rect[1]
    if rect_w <= 1 or rect_h <= 1:
        return green_crop, red_crop

    angle = float(rect[2])
    if rect_w < rect_h:
        angle += 90.0
    if angle > 45.0:
        angle -= 90.0
    if angle < -45.0:
        angle += 90.0

    no_deskew_std = _scan_angle_std(green_crop, red_crop)

    best_green = green_crop
    best_red = red_crop
    best_std = no_deskew_std

    # 先尝试让扫描条带与颜色带上下边界平行；只有稳定性明显变好时才启用。
    boundary_angle = _estimate_band_boundary_angle(green_crop, red_crop)
    if boundary_angle is not None and 3.0 <= abs(boundary_angle) <= 25.0:
        boundary_green, boundary_red = _rotate_mask_pair(green_crop, red_crop, boundary_angle)
        boundary_std = _scan_angle_std(boundary_green, boundary_red)
        if boundary_std is not None and (best_std is None or boundary_std < best_std * 0.85):
            best_green = boundary_green
            best_red = boundary_red
            best_std = boundary_std

    # 如果不旋转或边界平行处理后已经稳定，就不再强行做 minAreaRect deskew。
    if best_std is not None and best_std <= 1.0:
        return best_green, best_red

    # 只处理轻微倾斜；角度过大通常来自外部杂乱色块，不应强行旋转。
    if abs(angle) < 2.0 or abs(angle) > 25.0:
        return best_green, best_red

    rect_green, rect_red = _rotate_mask_pair(green_crop, red_crop, angle)
    rect_std = _scan_angle_std(rect_green, rect_red)
    if rect_std is not None and (best_std is None or rect_std < best_std * 0.85):
        return rect_green, rect_red

    return best_green, best_red


def _measure_color_widths(
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    band_box: tuple[int, int, int, int],
) -> tuple[int, int, Optional[float]]:
    """在多条水平扫描线上测宽，并逐线计算角度后取中位数。"""
    x, y, w, h = band_box
    if w <= 0 or h <= 0:
        return 0, 0, None

    x1 = max(0, x)
    x2 = min(green_mask.shape[1], x + w)
    y0 = max(0, y)
    y3 = min(green_mask.shape[0], y + h)
    if y3 <= y0 or x2 <= x1:
        return 0, 0, None

    green_roi = green_mask[y0:y3, x1:x2]
    red_roi = red_mask[y0:y3, x1:x2]
    green_roi, red_roi = _deskew_band_masks(green_roi, red_roi)
    initial_green_width, initial_red_width, initial_geometry_angle = _rough_measure_crop(green_roi, red_roi)
    if (
        initial_geometry_angle is not None
        and initial_geometry_angle >= 60.0
        and initial_green_width >= 200
        and 0 < initial_red_width <= 60
    ):
        green_roi = _fill_green_gap_after_red_crop(green_roi, red_roi)
    roi_h, roi_w = green_roi.shape[:2]
    # 避开顶部刻度和底部透明罩边缘，取中部偏宽一点的稳定区域。
    y1 = max(0, int(roi_h * 0.30))
    y2 = min(roi_h, int(roi_h * 0.70))
    if y2 <= y1 or roi_w <= 0:
        return 0, 0, None

    row_measurements = []
    # 多取一些扫描线，再过滤掉总宽度异常的行，降低镜头畸变、弯曲边缘和局部反光的影响。
    sample_count = min(31, max(7, y2 - y1))
    sample_rows = np.linspace(y1, y2 - 1, sample_count).astype(int)

    for row in sample_rows:
        # 每条扫描线分别寻找红/绿连续段，取该行最大的连续色段作为当前行宽度。
        green_runs = _column_runs(np.where(green_roi[row, :] > 0)[0])
        red_runs = _column_runs(np.where(red_roi[row, :] > 0)[0])

        green_width = max((end - start + 1 for start, end in green_runs), default=0)
        red_width = max((end - start + 1 for start, end in red_runs), default=0)

        # 每条扫描线上的红绿宽度来自同一条水平截线，先单独换算角度更符合几何关系。
        line_angle = _projection_calibration_from_widths(green_width, red_width) if green_width > 0 and red_width > 0 else None
        if line_angle is not None:
            row_measurements.append((green_width, red_width, green_width + red_width, line_angle))

    if not row_measurements:
        return 0, 0, None

    rows = np.array(row_measurements, dtype=np.float32)
    # 镜头畸变或 ROI 边缘弯曲时，部分扫描线的红绿总投影宽度会异常偏大/偏小。
    # 过滤总宽度两端分位后再取中位数，能保留主体色带测量，同时避开边缘畸变行。
    if len(rows) >= 5:
        low, high = np.percentile(rows[:, 2], [20, 80])
        filtered = rows[(rows[:, 2] >= low) & (rows[:, 2] <= high)]
        if len(filtered) >= 3:
            rows = filtered

    # 中位数比平均数更抗异常扫描线，例如文字边缘、反光、局部漏检。
    green_width = int(round(float(np.median(rows[:, 0]))))
    red_width = int(round(float(np.median(rows[:, 1]))))
    geometry_angle = float(np.median(rows[:, 3]))
    return green_width, red_width, geometry_angle


def _projection_calibration_from_widths(green_width: int, red_width: int) -> Optional[float]:
    """根据 PPT 中的圆弦几何关系，把侧面投影宽度换算为真实角度。"""
    total_width = green_width + red_width
    if total_width <= 0:
        return None

    # 红绿总开度固定为 80 度，w_red + w_green 是这段圆弧对应的投影弦长。
    # (w1 + w2)^2 = 2r^2 - 2r^2*cos(80°)
    theta_total = math.radians(80.0)
    radius = total_width / math.sqrt(2.0 - 2.0 * math.cos(theta_total))

    # 将红绿分界点看作总弦上的一个点，计算它相对弦中心的偏移，
    # 再用 asin 将投影偏移换算成相对 40 度中线的角度偏移。
    split_x = green_width - total_width / 2.0
    ratio = max(-1.0, min(1.0, split_x / radius))
    angle = 40.0 + math.degrees(math.asin(ratio))
    return min(max(angle, 0.0), 80.0)


def _main_color_band(green_mask: np.ndarray, red_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """定位侧面彩色指示区域 ROI，优先选择同时包含红绿两侧的候选框。"""
    combined = cv2.bitwise_or(green_mask, red_mask)
    h, w = combined.shape[:2]
    if int(np.count_nonzero(combined)) == 0:
        return np.zeros_like(combined), (0, 0, 0, 0)

    candidates: list[tuple[float, int, int, int, int]] = []

    def add_candidate(x: int, y: int, bw: int, bh: int, source_weight: float = 0.0) -> None:
        """给一个候选 ROI 打分，红绿都存在且能测到两侧宽度的框优先。"""
        if bw <= 0 or bh <= 0:
            return

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        bw = x2 - x
        bh = y2 - y
        if bw <= 0 or bh <= 0:
            return

        roi_green = green_mask[y:y2, x:x2]
        roi_red = red_mask[y:y2, x:x2]
        green_pixels = int(np.count_nonzero(roi_green))
        red_pixels = int(np.count_nonzero(roi_red))
        total_pixels = green_pixels + red_pixels
        if total_pixels == 0:
            return

        green_width, red_width, geometry_angle = _measure_color_widths(green_mask, red_mask, (x, y, bw, bh))
        has_both_pixels = green_pixels > 0 and red_pixels > 0
        has_both_widths = green_width > 0 and red_width > 0 and geometry_angle is not None
        # 侧面指示带通常是一条红绿相邻的横向区域。这里把红绿共存放在最高优先级，
        # 避免 ROI 被单独的大红块、大绿块或背景反光色块带偏。
        balance_score = min(green_pixels, red_pixels) * 2.0
        width_score = min(green_width, red_width) * 8_000.0
        total_score = total_pixels * 0.05
        both_bonus = 2_000_000.0 if has_both_pixels else 0.0
        width_bonus = 1_000_000.0 if has_both_widths else 0.0
        aspect = bw / max(float(bh), 1.0)
        aspect_bonus = 80_000.0 if 0.9 <= aspect <= 5.5 else 0.0
        height_ratio = bh / max(float(h), 1.0)
        tall_square_penalty = max(0.0, height_ratio - 0.38) * 5_000_000.0 if aspect < 1.35 else 0.0
        score = (
            source_weight
            + both_bonus
            + width_bonus
            + balance_score
            + width_score
            + total_score
            + aspect_bonus
            - tall_square_penalty
        )
        candidates.append((score, x, y, x2, y2))

    min_area = max(120.0, h * w * 0.00025)

    # 候选 1：红绿轮廓配对。红绿区域经常会被文字、黑边或反光拆成两个连通块，
    # 所以不直接取最大色块，而是先找垂直方向重叠、水平距离接近的一红一绿组合。
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_boxes = []
    red_boxes = []
    red_pair_min_area = min_area * 0.45
    for cnt in green_contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            green_boxes.append((*cv2.boundingRect(cnt), area))
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area >= red_pair_min_area:
            red_boxes.append((*cv2.boundingRect(cnt), area))

    for gx, gy, gw, gh, _ in green_boxes:
        for rx, ry, rw, rh, _ in red_boxes:
            y_overlap = max(0, min(gy + gh, ry + rh) - max(gy, ry))
            x_gap = max(0, max(gx, rx) - min(gx + gw, rx + rw))
            vertical_ok = y_overlap >= min(gh, rh) * 0.12
            horizontal_ok = x_gap <= max(gw, rw) * 1.6
            if not (vertical_ok and horizontal_ok):
                continue

            x1 = min(gx, rx)
            y1 = min(gy, ry)
            x2 = max(gx + gw, rx + rw)
            y2 = max(gy + gh, ry + rh)
            pad_x = max(3, int((x2 - x1) * 0.04))
            pad_y = max(3, int((y2 - y1) * 0.12))
            add_candidate(x1 - pad_x, y1 - pad_y, (x2 - x1) + 2 * pad_x, (y2 - y1) + 2 * pad_y, 400_000.0)

    # 候选 2：行列投影。适合红绿已经比较连续、同处一条横向色带的图片。
    projected = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((7, 21), np.uint8))
    row_counts = np.sum(projected > 0, axis=1)
    if row_counts.size and int(row_counts.max()) > 0:
        row_threshold = max(5, int(row_counts.max() * 0.18))
        row_runs = _column_runs(np.where(row_counts >= row_threshold)[0])
        for y_start, y_end in row_runs:
            band = projected[y_start : y_end + 1, :]
            band_h = y_end - y_start + 1
            col_counts = np.sum(band > 0, axis=0)
            col_threshold = max(2, int(band_h * 0.08))
            col_runs = _column_runs(np.where(col_counts >= col_threshold)[0])
            for x_start, x_end in col_runs:
                pad_x = max(2, int((x_end - x_start + 1) * 0.02))
                pad_y = max(2, int((y_end - y_start + 1) * 0.15))
                add_candidate(
                    x_start - pad_x,
                    y_start - pad_y,
                    (x_end - x_start + 1) + 2 * pad_x,
                    (y_end - y_start + 1) + 2 * pad_y,
                    120_000.0,
                )

    # 候选 3：最大连通块兜底。极端角度下可能只有一侧颜色明显，仍然要能返回 ROI。
    contours, _ = cv2.findContours(projected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        pad_x = max(2, int(bw * 0.02))
        pad_y = max(2, int(bh * 0.10))
        add_candidate(x - pad_x, y - pad_y, bw + 2 * pad_x, bh + 2 * pad_y)

    if not candidates:
        return np.zeros_like(combined), (0, 0, 0, 0)

    _, x, y, x2, y2 = max(candidates, key=lambda item: item[0])
    band_mask = np.zeros_like(combined)
    band_mask[y:y2, x:x2] = 255
    return band_mask, (x, y, x2 - x, y2 - y)


def predict_side_cv(image_path: str, calibrate: str = "auto", return_debug: bool = False):
    """预测单张侧面图片的阀门开度角。"""
    img = _read_image(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    img = _enhance_if_dark(img)

    # 1. 红绿颜色初筛。
    green_mask_raw = _filter_green(img)
    red_mask_raw = _filter_red(img)

    # 2. 像素补全：填充文字造成的空洞，再用于后续面积和宽度计算。
    green_mask = _complete_color_mask(green_mask_raw)
    green_mask = _recover_nearby_green_fragments(green_mask_raw, green_mask)
    red_mask = _complete_color_mask(red_mask_raw)
    red_mask = _recover_left_red_sliver(red_mask_raw, red_mask, green_mask)
    red_mask = _keep_red_adjacent_to_green(green_mask, red_mask)

    # 3. 定位主颜色带，并用多条水平扫描线测量红绿投影宽度。
    band_mask, band_box = _main_color_band(green_mask, red_mask)
    green_mask = _recover_green_highlight_in_roi(img, green_mask, red_mask, band_box)
    initial_green_width, initial_red_width, initial_geometry_angle = _measure_color_widths(
        green_mask, red_mask, band_box
    )
    # 右扫补全只用于绿色被反光低估到 20 多度的场景；低角度或高角度样本直接补到右边界会明显过补。
    if (
        initial_geometry_angle is not None
        and 22.0 <= initial_geometry_angle <= 26.8
        and 225 <= initial_red_width <= 280
        and initial_green_width <= 120
    ):
        repair_box = (
            band_box[0],
            band_box[1],
            band_box[2] + max(20, int(round(band_box[2] * 0.08))),
            band_box[3],
        )
        green_mask = _recover_green_right_scan_in_roi(
            green_mask, red_mask, repair_box, green_mask_raw, allow_edge_only=True
        )
    if (
        initial_geometry_angle is not None
        and initial_geometry_angle >= 60.0
        and initial_green_width >= 200
        and 0 < initial_red_width <= 60
    ):
        green_mask = _recover_green_gap_after_red_in_roi(green_mask, red_mask, band_box)
    red_mask = _keep_red_adjacent_to_green(green_mask, red_mask)
    band_mask, band_box = _main_color_band(green_mask, red_mask)
    green_width, red_width, geometry_angle = _measure_color_widths(green_mask, red_mask, band_box)

    # raw_angle 是像素面积占比估计，保留用于兜底和调试，不作为默认输出。
    green_in_band = cv2.bitwise_and(green_mask, band_mask)
    red_in_band = cv2.bitwise_and(red_mask, band_mask)
    green_pixels = int(np.count_nonzero(green_in_band))
    red_pixels = int(np.count_nonzero(red_in_band))
    total_pixels = green_pixels + red_pixels

    if total_pixels == 0:
        angle = 0.0
        raw_angle = 0.0
    else:
        raw_angle = green_pixels / total_pixels * 80.0
        use_calibration = calibrate in {"auto", "always"}
        # 默认以逐扫描线几何角度为主；当面积角与几何角差距不大时，混入少量面积角修正系统偏差。
        if use_calibration and geometry_angle is not None:
            raw_weight = 0.0 if abs(raw_angle - geometry_angle) > 14.1 else 0.33
            angle = (1.0 - raw_weight) * geometry_angle + raw_weight * raw_angle
        else:
            angle = raw_angle

    angle = round(min(max(angle, 0.0), 80.0), 1)
    use_calibration = calibrate in {"auto", "always"}
    debug = SidePredictionDebug(
        raw_angle=round(raw_angle, 3),
        calibrated=use_calibration and geometry_angle is not None,
        geometry_angle=round(geometry_angle, 3) if geometry_angle is not None else None,
        green_width=green_width,
        red_width=red_width,
        green_pixels=green_pixels,
        red_pixels=red_pixels,
        band_box=band_box,
    )
    return (angle, debug) if return_debug else angle


def predict_cvnew(image_path: str) -> float:
    """项目统一入口：使用最新侧面 CVnew 流程预测单张图片。"""
    return predict_side_cv(image_path)


def parse_angle_from_filename(filename: str) -> Optional[float]:
    """从文件名中解析真实角度，例如 0029_23.1.jpg -> 23.1。"""
    match = ANGLE_RE.search(filename)
    return float(match.group(1)) if match else None


def iter_images(folder_path: str) -> Iterable[str]:
    """按文件名顺序遍历文件夹中的图片。"""
    for name in sorted(os.listdir(folder_path)):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            yield os.path.join(folder_path, name)


def evaluate_folder(folder_path: str, output_csv: Optional[str] = None, calibrate: str = "auto"):
    """批量评估文件夹中的侧面图片，并可输出逐图 CSV。"""
    rows = []
    for image_path in iter_images(folder_path):
        filename = os.path.basename(image_path)
        gt = parse_angle_from_filename(filename)
        if gt is None:
            continue
        pred, debug = predict_side_cv(image_path, calibrate=calibrate, return_debug=True)
        err = abs(pred - gt)
        rows.append(
            {
                "filename": filename,
                "ground_truth": gt,
                "prediction": pred,
                "abs_error": round(err, 3),
                "raw_angle": debug.raw_angle,
                "geometry_angle": debug.geometry_angle,
                "calibrated": debug.calibrated,
                "green_width": debug.green_width,
                "red_width": debug.red_width,
                "green_pixels": debug.green_pixels,
                "red_pixels": debug.red_pixels,
                "band_box": debug.band_box,
            }
        )

    errors = [r["abs_error"] for r in rows]
    summary = {
        "count": len(rows),
        "mae": round(float(np.mean(errors)), 3) if errors else 0.0,
        "rmse": round(float(math.sqrt(np.mean(np.square(errors)))), 3) if errors else 0.0,
        "max_error": round(float(np.max(errors)), 3) if errors else 0.0,
    }

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["filename"])
            writer.writeheader()
            writer.writerows(rows)

    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-view OpenCV valve angle predictor")
    parser.add_argument("--input", required=True, help="Image folder")
    parser.add_argument("--output", default="output/side_cv_results.csv", help="Output CSV path")
    parser.add_argument(
        "--calibrate",
        choices=["auto", "always", "never"],
        default="auto",
        help="Apply side projection calibration",
    )
    args = parser.parse_args()

    summary, rows = evaluate_folder(args.input, args.output, args.calibrate)
    print(f"count: {summary['count']}")
    print(f"MAE: {summary['mae']}")
    print(f"RMSE: {summary['rmse']}")
    print(f"Max error: {summary['max_error']}")
    print(f"Saved: {args.output}")

    worst = sorted(rows, key=lambda r: r["abs_error"], reverse=True)[:10]
    if worst:
        print("Worst 10:")
        for row in worst:
            print(
                f"{row['filename']}: gt={row['ground_truth']:.1f}, "
                f"pred={row['prediction']:.1f}, err={row['abs_error']:.1f}, "
                f"raw={row['raw_angle']:.1f}, calibrated={row['calibrated']}"
            )


if __name__ == "__main__":
    main()
