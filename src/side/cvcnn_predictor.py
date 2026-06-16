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


def _crop_valve_region(
    img: np.ndarray,
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    pad_ratio: float = 0.25,
) -> np.ndarray:
    """根据红绿颜色带定位阀门区域并裁剪。"""
    band_mask, band_box = _main_color_band(green_mask, red_mask)
    x, y, bw, bh = band_box

    if bw <= 0 or bh <= 0:
        # 检测失败，返回原图
        return img

    # 用颜色带重新定位，补回高光
    green_mask = _recover_green_highlight_in_roi(img, green_mask, red_mask, band_box)
    _, band_box = _main_color_band(green_mask, red_mask)
    x, y, bw, bh = band_box

    if bw <= 0 or bh <= 0:
        return img

    h, w = img.shape[:2]

    # 加 padding
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

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
