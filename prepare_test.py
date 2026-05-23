import os
import random
import shutil
import argparse

def prepare_test(input_dir, output_dir="test_input", count=10):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected = random.sample(files, min(count, len(files)))

    for i, f in enumerate(selected, 1):
        ext = os.path.splitext(f)[1]
        shutil.copy2(os.path.join(input_dir, f), os.path.join(output_dir, f"{i}.png"))
        print(f"  {i}.png <- {f}")

    print(f"\n已复制 {len(selected)} 张到 {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="unlabbled_top", help="源图片目录")
    parser.add_argument("--output", default="test_input", help="输出目录")
    parser.add_argument("--count", type=int, default=10, help="选取数量")
    args = parser.parse_args()

    prepare_test(args.input, args.output, args.count)
