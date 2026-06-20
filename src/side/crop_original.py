"""
对原始 side 图片做 CV 裁剪，提取阀门区域
流程：遍历 origin data/side/ → _crop_valve_region() → 保存到 data_cropped_original/side/
裁剪失败则跳过
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np

from src.side.cvnew_predictor import (
    _read_image, _enhance_if_dark, _filter_green, _filter_red,
    _complete_color_mask, _recover_nearby_green_fragments,
    _recover_left_red_sliver, _keep_red_adjacent_to_green,
    _main_color_band, _recover_green_highlight_in_roi,
)
from src.side.cvcnn_predictor import _crop_valve_region


def crop_single_image(img_path: str) -> np.ndarray | None:
    """对单张图片执行 CV 裁剪，返回裁剪后的 BGR 图或 None"""
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

    if cropped.shape[:2] == img.shape[:2]:
        return None

    return cropped


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CV 裁剪原始 side 图片')
    parser.add_argument('--input', default='origin data/side', help='原始数据目录')
    parser.add_argument('--output', default='data_cropped_original/side', help='裁剪输出目录')
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
        cropped = crop_single_image(img_path)

        if cropped is None:
            skip += 1
            print(f"  跳过: {f}")
            continue

        out_path = os.path.join(output_dir, f)
        cv2.imencode('.jpg', cropped)[1].tofile(out_path)
        success += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}] 成功 {success}, 跳过 {skip}")

    print(f"\n裁剪完成！成功 {success} 张，跳过 {skip} 张")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
