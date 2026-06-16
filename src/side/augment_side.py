"""Side 视角数据增强脚本"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.common.data_augment import augment_dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Side 视角数据增强')
    parser.add_argument('--augment', type=int, default=95, help='每张图增强次数 (默认 95)')
    parser.add_argument('--source', default='origin data/side', help='源数据目录')
    parser.add_argument('--output', default=None, help='输出目录（默认 data_augmented/<源末级目录>/）')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(project_root, args.source)

    if args.output:
        output_dir = os.path.join(project_root, args.output)
    else:
        view_name = os.path.basename(os.path.normpath(args.source))
        output_dir = os.path.join(os.path.dirname(data_dir), 'data_augmented', view_name)

    files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"数据源: {args.source}/ ({len(files)} 张)")
    print(f"增强倍数: {args.augment}，预计产出: {len(files) * (args.augment + 1)} 张")
    print(f"输出目录: {output_dir}\n")

    augmented_files = augment_dataset(data_dir, output_dir, args.augment)
    print(f"\n增强完成！共 {len(augmented_files)} 张")


if __name__ == '__main__':
    main()
