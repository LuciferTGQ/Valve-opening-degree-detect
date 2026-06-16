import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import pandas as pd
from typing import List, Tuple

from src.common.ensemble import predict_ensemble
from src.top.cvnew_predictor import predict_cvnew
from src.common.cnn_predictor import predict_cnn

def predict_folder(
    folder_path: str,
    output_csv: str,
    model_path: str = None,
    use_cnn: bool = True
) -> List[Tuple[str, float]]:
    """
    预测文件夹中所有图片的角度

    Args:
        folder_path: 图片文件夹路径
        output_csv: 输出 CSV 路径
        model_path: CNN 模型路径
        use_cnn: 是否使用 CNN

    Returns:
        [(filename, angle), ...]
    """
    # 1. 获取所有图片
    image_files = []
    for f in sorted(os.listdir(folder_path)):
        if f.endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(f)

    if not image_files:
        print(f"文件夹中没有图片: {folder_path}")
        return []

    print(f"找到 {len(image_files)} 张图片")

    # 2. 逐张预测
    results = []

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)

        if use_cnn and model_path and os.path.exists(model_path):
            # 使用融合预测
            angle = predict_ensemble(image_path, model_path)
            method = "ensemble"
        else:
            # 仅使用 CVnew
            angle = predict_cvnew(image_path)
            method = "cvnew"

        results.append((filename, angle))
        print(f"{filename}: {angle}° ({method})")

    # 3. 保存 CSV
    df = pd.DataFrame(results, columns=['filename', 'angle'])
    df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到: {output_csv}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='预测阀门开度角度')
    parser.add_argument('--input', type=str, required=True, help='输入图片文件夹')
    parser.add_argument('--output', type=str, default='output/result.csv', help='输出 CSV 路径')
    parser.add_argument('--model', type=str, default='models/mobilenetv3_top.pth', help='CNN 模型路径')
    parser.add_argument('--no-cnn', action='store_true', help='不使用 CNN，仅用 CV')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    predict_folder(
        folder_path=args.input,
        output_csv=args.output,
        model_path=args.model,
        use_cnn=not args.no_cnn
    )
