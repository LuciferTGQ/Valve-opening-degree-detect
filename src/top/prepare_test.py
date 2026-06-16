"""准备 top 视角测试数据"""

import os
import sys
import random
import shutil
import csv
import argparse


def prepare_unlabeled(input_dir, output_dir, count=10):
    """从未标注数据中随机采样（无 ground truth）"""
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected = random.sample(files, min(count, len(files)))

    for i, f in enumerate(selected, 1):
        ext = os.path.splitext(f)[1]
        shutil.copy2(os.path.join(input_dir, f), os.path.join(output_dir, f"{i}.png"))
        print(f"  {i}.png <- {f}")

    print(f"\n已复制 {len(selected)} 张到 {output_dir}/")


def prepare_labeled(input_dir, output_dir, gt_csv, count=10):
    """从已标注数据中随机采样（生成 ground truth CSV）"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(gt_csv), exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected = random.sample(files, min(count, len(files)))

    gt_rows = []
    for i, f in enumerate(selected, 1):
        shutil.copy2(os.path.join(input_dir, f), os.path.join(output_dir, f"{i}.png"))
        angle = float(f.split('_')[1].replace('.jpg', '').replace('.png', ''))
        gt_rows.append((f"{i}.png", angle))
        print(f"  {i}.png <- {f}  (角度: {angle}°)")

    with open(gt_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['filename', 'angle'])
        writer.writerows(gt_rows)

    print(f"\n已复制 {len(selected)} 张到 {output_dir}/")
    print(f"标准答案已保存到 {gt_csv}")


def main():
    parser = argparse.ArgumentParser(description='准备 top 视角测试数据')
    parser.add_argument('--mode', choices=['labeled', 'unlabeled'], default='labeled',
                        help='labeled: 从已标注数据采样(带GT); unlabeled: 从未标注数据采样')
    parser.add_argument('--input', default=None, help='输入目录（默认按 mode 自动选择）')
    parser.add_argument('--output', default='test_input_top', help='输出目录')
    parser.add_argument('--gt', default='output/ground_truth_top.csv', help='GT CSV 路径（labeled 模式）')
    parser.add_argument('--count', type=int, default=10, help='采样数量')
    args = parser.parse_args()

    if args.mode == 'unlabeled':
        input_dir = args.input or 'unlabbled_top'
        prepare_unlabeled(input_dir, args.output, args.count)
    else:
        input_dir = args.input or 'origin data/top'
        prepare_labeled(input_dir, args.output, args.gt, args.count)


if __name__ == '__main__':
    main()
