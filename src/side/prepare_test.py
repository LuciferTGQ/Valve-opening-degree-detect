"""准备 side 视角测试数据"""

import os
import sys
import random
import shutil
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def prepare_labeled(input_dir, output_dir, gt_csv, count=10):
    """从已标注数据中随机采样（生成 ground truth CSV）"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(gt_csv), exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    selected = random.sample(files, min(count, len(files)))

    gt_rows = []
    for i, f in enumerate(selected, 1):
        ext = os.path.splitext(f)[1]
        new_name = f"{i}{ext}"
        shutil.copy2(os.path.join(input_dir, f), os.path.join(output_dir, new_name))
        angle = float(f.split('_')[1].replace('.jpg', '').replace('.png', ''))
        gt_rows.append((new_name, angle))
        print(f"  {new_name} <- {f}  (角度: {angle}°)")

    with open(gt_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['filename', 'angle'])
        writer.writerows(gt_rows)

    print(f"\n已复制 {len(selected)} 张到 {output_dir}/")
    print(f"标准答案已保存到 {gt_csv}")


def prepare_unlabeled(input_dir, output_dir, count=10):
    """从未标注数据中随机采样（无 ground truth）"""
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected = random.sample(files, min(count, len(files)))

    for i, f in enumerate(selected, 1):
        ext = os.path.splitext(f)[1]
        shutil.copy2(os.path.join(input_dir, f), os.path.join(output_dir, f"{i}{ext}"))
        print(f"  {i}{ext} <- {f}")

    print(f"\n已复制 {len(selected)} 张到 {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='准备 side 视角测试数据')
    parser.add_argument('--mode', choices=['labeled', 'unlabeled'], default='labeled')
    parser.add_argument('--input', default=None, help='输入目录（默认按 mode 自动选择）')
    parser.add_argument('--output', default='test_input_side', help='输出目录')
    parser.add_argument('--gt', default='output/ground_truth_side.csv', help='GT CSV 路径')
    parser.add_argument('--count', type=int, default=10, help='采样数量')
    args = parser.parse_args()

    if args.mode == 'unlabeled':
        input_dir = args.input or 'unlabbled_side'
        prepare_unlabeled(input_dir, args.output, args.count)
    else:
        input_dir = args.input or 'origin data/side'
        prepare_labeled(input_dir, args.output, args.gt, args.count)


if __name__ == '__main__':
    main()
