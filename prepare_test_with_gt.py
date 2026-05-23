import os
import random
import shutil
import csv
import argparse

def prepare_test_with_gt(input_dir="origin data/top", output_dir="test_input", gt_csv="output/ground_truth.csv", count=10):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="origin data/top", help="源图片目录")
    parser.add_argument("--output", default="test_input", help="输出目录")
    parser.add_argument("--gt", default="output/ground_truth.csv", help="标准答案CSV路径")
    parser.add_argument("--count", type=int, default=10, help="选取数量")
    args = parser.parse_args()

    prepare_test_with_gt(args.input, args.output, args.gt, args.count)
