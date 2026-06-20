"""
Side 视角 CV+CNN 预测器
CV 提取红绿色块区域 → 裁剪阀门区域 → 喂给 CNN 推理。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.common.cnn_predictor import ValveAngleRegressor, get_transform
from src.side.cvnew_predictor import (
    _read_image,
    _enhance_if_dark,
    _filter_green,
    _filter_red,
    _complete_color_mask,
    _recover_nearby_green_fragments,
    _recover_left_red_sliver,
    _keep_red_adjacent_to_green,
    _main_color_band,
    _recover_green_highlight_in_roi,
    _measure_color_widths,
)


def _locate_red_only_band(red_mask: np.ndarray, green_mask: np.ndarray) -> tuple:
    """当绿色几乎不存在时，用红色行投影定位阀门色带区域。

    先在红色 mask 中只保留最密集水平行带内的像素，过滤背景散布红点，
    再基于过滤后的 mask 定位边界框。
    """
    h, w = red_mask.shape[:2]
    if np.count_nonzero(red_mask) == 0:
        return (0, 0, 0, 0)

    # 行投影：找红色像素最密集的水平行带
    row_counts = np.sum(red_mask > 0, axis=1)
    if row_counts.max() == 0:
        return (0, 0, 0, 0)

    row_threshold = max(5, int(row_counts.max() * 0.30))
    row_active = np.where(row_counts >= row_threshold)[0]
    if len(row_active) == 0:
        return (0, 0, 0, 0)

    # 找最长连续行段（阀门色带在垂直方向是连续的）
    runs = []
    start = row_active[0]
    for i in range(1, len(row_active)):
        if row_active[i] - row_active[i - 1] > 3:
            runs.append((start, row_active[i - 1]))
            start = row_active[i]
    runs.append((start, row_active[-1]))

    # 选密度最高的行段（平均行像素数最多）
    best_run = max(runs, key=lambda r: row_counts[r[0]:r[1] + 1].sum() / max(r[1] - r[0] + 1, 1))
    y_start, y_end = best_run

    # 只保留该行段内的红色像素，过滤行段外的背景红点
    filtered_red = np.zeros_like(red_mask)
    filtered_red[y_start:y_end + 1, :] = red_mask[y_start:y_end + 1, :]

    # 行段内再做列过滤：只保留列像素密度足够的列
    band_red = filtered_red[y_start:y_end + 1, :]
    col_counts = np.sum(band_red > 0, axis=0)
    band_h = y_end - y_start + 1
    col_threshold = max(3, int(band_h * 0.15))
    col_active = np.where(col_counts >= col_threshold)[0]
    if len(col_active) == 0:
        return (0, 0, 0, 0)

    # 列范围：取连续列段的90%分位避免极端边缘
    x_start = int(col_active[0])
    x_end = int(col_active[-1])

    bw = x_end - x_start + 1
    bh = y_end - y_start + 1

    if bw <= 0 or bh <= 0:
        return (0, 0, 0, 0)

    return (x_start, y_start, bw, bh)


def _crop_valve_region(
    img: np.ndarray,
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    pad_ratio: float = 0.25,
) -> np.ndarray:
    """根据红绿颜色带定位阀门区域并裁剪。

    当绿色几乎不存在（低角度）时，回退到红色行投影定位。
    highlight recovery 只用于角度测量，裁剪时不用以免定位变差。
    """
    h, w = img.shape[:2]
    img_area = h * w
    green_ratio = np.count_nonzero(green_mask) / img_area
    red_ratio = np.count_nonzero(red_mask) / img_area

    # 绿色极少时用红色行投影回退
    use_red_only = green_ratio < 0.001 and red_ratio > 0.0001

    if use_red_only:
        band_box = _locate_red_only_band(red_mask, green_mask)
        x, y, bw, bh = band_box
        if bw <= 0 or bh <= 0:
            return img
    else:
        band_mask, band_box = _main_color_band(green_mask, red_mask)
        x, y, bw, bh = band_box

        if bw <= 0 or bh <= 0:
            return img

        # 校验裁剪区域是否合理（过大说明定位失败）
        crop_area = bw * bh
        if crop_area > img_area * 0.8:
            # 定位失败，尝试红色行投影回退
            band_box = _locate_red_only_band(red_mask, green_mask)
            x, y, bw, bh = band_box
            if bw <= 0 or bh <= 0:
                return img

    # 加 padding
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    # 裁剪区域最小高度：确保不会裁成极窄横条
    min_crop_h = max(bh, int(h * 0.12))
    pad_y = max(pad_y, (min_crop_h - bh) // 2)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img
    return cropped


def predict_cvcnn(image_path: str, model_path: str, pad_ratio: float = 0.25) -> float:
    """
    CV+CNN 预测：先用 CV 提取阀门区域，再用 CNN 推理角度。

    Args:
        image_path: 图片路径
        model_path: CNN 模型路径
        pad_ratio: 裁剪区域的 padding 比例

    Returns:
        预测角度 (0-80)
    """
    # 1. CV 预处理：提取红绿色块区域
    img = _read_image(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    img = _enhance_if_dark(img)
    green_mask_raw = _filter_green(img)
    red_mask_raw = _filter_red(img)

    green_mask = _complete_color_mask(green_mask_raw)
    green_mask = _recover_nearby_green_fragments(green_mask_raw, green_mask)
    red_mask = _complete_color_mask(red_mask_raw)
    red_mask = _recover_left_red_sliver(red_mask_raw, red_mask, green_mask)
    red_mask = _keep_red_adjacent_to_green(green_mask, red_mask)

    # 2. 裁剪阀门区域
    cropped = _crop_valve_region(img, green_mask, red_mask, pad_ratio)

    # 3. CNN 推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ValveAngleRegressor()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    transform = get_transform()
    # BGR → RGB → PIL
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cropped_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        angle = model(tensor).item()

    return round(min(max(angle, 0.0), 80.0), 1)
