"""
对裁剪后的 side 图片按角度均衡增强

统计每个角度的原始图片数量，少的增强多、多的增强少，
目标每个角度达到 ~240 张，总量约 2 万。
"""

import os
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np
from collections import defaultdict

from src.common.data_augment import augment_image

ANGLE_RE = re.compile(r'_([0-9]+(?:\.[0-9]+)?)\.(?:jpg|jpeg|png)$', re.IGNORECASE)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='按角度均衡增强裁剪后的 side 图片')
    parser.add_argument('--input', default='data_cropped_original/side', help='裁剪数据目录')
    parser.add_argument('--output', default='data_augmented_cropped/side', help='增强输出目录')
    parser.add_argument('--target', type=int, default=240, help='每个角度目标数量')
    parser.add_argument('--max-augment', type=int, default=40, help='单张图片最大增强次数')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_dir = os.path.join(project_root, args.input)
    output_dir = os.path.join(project_root, args.output)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 统计每个角度的图片数量
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    angle_files = defaultdict(list)
    for f in files:
        m = ANGLE_RE.search(f)
        if m:
            angle = float(m.group(1))
            angle_files[angle].append(f)

    print(f"输入: {len(files)} 张裁剪图片, {len(angle_files)} 个角度")
    print(f"目标: 每角度 ~{args.target} 张, 单图最多增强 {args.max_augment} 次\n")

    # 2. 计算每个角度的增强倍数
    plan = {}
    total_target = 0
    for angle, file_list in sorted(angle_files.items()):
        n = len(file_list)
        needed = args.target - n
        if needed <= 0:
            augment_per_img = 0
            actual = n
        else:
            augment_per_img = min(args.max_augment, needed // n + (1 if needed % n else 0))
            actual = n + n * augment_per_img
        plan[angle] = (file_list, augment_per_img, actual)
        total_target += actual

    print(f"增强计划:")
    for angle, (file_list, aug, actual) in sorted(plan.items()):
        print(f"  {angle:>6.1f}°: {len(file_list)}张, 增强{aug}x → {actual}张")
    print(f"\n预计总量: {total_target} 张\n")

    # 3. 执行增强
    success = 0
    for angle, (file_list, augment_per_img, actual) in sorted(plan.items()):
        for f in file_list:
            img_path = os.path.join(input_dir, f)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            base_name = os.path.splitext(f)[0]

            # 保存原图
            out_path = os.path.join(output_dir, f"{base_name}_original.jpg")
            cv2.imencode('.jpg', img)[1].tofile(out_path)
            success += 1

            # 增强
            for i in range(augment_per_img):
                augmented = augment_image(img, seed=i)
                aug_path = os.path.join(output_dir, f"{base_name}_aug{i:02d}.jpg")
                cv2.imencode('.jpg', augmented)[1].tofile(aug_path)
                success += 1

        print(f"  {angle:>6.1f}°: 完成 ({actual}张)")

    print(f"\n增强完成！共 {success} 张")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
