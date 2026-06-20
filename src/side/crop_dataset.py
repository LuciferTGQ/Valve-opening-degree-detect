"""
Side 视角裁剪数据集
训练时动态用 CV 裁剪阀门区域，然后喂给 CNN
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.common.data_augment import parse_angle_from_filename
from src.side.cvnew_predictor import (
    _read_image, _enhance_if_dark, _filter_green, _filter_red,
    _complete_color_mask, _recover_nearby_green_fragments,
    _recover_left_red_sliver, _keep_red_adjacent_to_green,
    _main_color_band, _recover_green_highlight_in_roi,
)
from src.side.cvcnn_predictor import _crop_valve_region


def _crop_image(img_path: str) -> Image.Image | None:
    """读取完整图片 → CV裁剪阀门区域 → 返回裁剪后的 PIL 图"""
    img = _read_image(img_path)
    if img is None:
        return None

    img = _enhance_if_dark(img)
    green_mask_raw = _filter_green(img)
    red_mask_raw = _filter_red(img)

    green_mask = _complete_color_mask(green_mask_raw)
    green_mask = _recover_nearby_green_fragments(green_mask_raw, green_mask)
    red_mask = _complete_color_mask(red_mask_raw)
    red_mask = _recover_left_red_sliver(red_mask_raw, red_mask, green_mask)
    red_mask = _keep_red_adjacent_to_green(green_mask, red_mask)

    cropped = _crop_valve_region(img, green_mask, red_mask)

    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)


class CropValveDataset(Dataset):
    """训练时动态裁剪阀门区域的数据集"""

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]

        # CV 裁剪
        cropped = _crop_image(img_path)
        if cropped is None:
            cropped = Image.open(img_path).convert('RGB')

        # 解析角度
        angle = parse_angle_from_filename(os.path.basename(img_path))

        if self.transform:
            cropped = self.transform(cropped)

        return cropped, torch.tensor([angle], dtype=torch.float32)
