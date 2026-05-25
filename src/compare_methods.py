"""
对比 CVnew、CNN 两种方法的精度
输出 MAE、RMSE、最大误差
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cvnew_predictor import predict_cvnew

try:
    from cnn_predictor import predict_cnn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch 未安装，跳过 CNN 对比")


def load_ground_truth(data_dir):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))])
    gt = []
    for f in files:
        parts = f.split('_')
        angle = float(parts[1].replace('.jpg', '').replace('.png', ''))
        gt.append((f, angle))
    return gt


def calc_metrics(gts, preds):
    errors = [abs(g - p) for g, p in zip(gts, preds)]
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
    max_err = max(errors)
    return mae, rmse, max_err


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='origin data/top', help='数据目录')
    parser.add_argument('--model', default='models/mobilenetv3_top.pth', help='CNN 模型路径')
    args = parser.parse_args()

    data_dir = args.data
    model_path = args.model

    data = load_ground_truth(data_dir)
    gts = [d[1] for d in data]
    paths = [os.path.join(data_dir, d[0]) for d in data]

    print(f"加载 {len(data)} 张图片 from {data_dir}\n")

    # 运行各方法
    print("运行 CVnew...")
    cvnew_preds = [predict_cvnew(p) for p in paths]

    cnn_preds = None
    if HAS_TORCH and os.path.exists(model_path):
        print("运行 CNN...")
        cnn_preds = [predict_cnn(p, model_path) for p in paths]

    # 输出整体指标
    methods = {'CVnew': cvnew_preds}
    if cnn_preds:
        methods['CNN'] = cnn_preds

    print("\n" + "=" * 50)
    print("整体指标")
    print("=" * 50)
    print("{:<12} {:>6} {:>7} {:>8}".format("方法", "MAE", "RMSE", "最大误差"))
    print("-" * 36)
    for name, preds in methods.items():
        mae, rmse, max_err = calc_metrics(gts, preds)
        print("{:<12} {:>6.2f} {:>7.2f} {:>8.2f}".format(name, mae, rmse, max_err))

    # 分段指标
    print("\n" + "=" * 50)
    print("分段指标 (MAE)")
    print("=" * 50)
    ranges = [(0, 20), (20, 40), (40, 60), (60, 80)]
    print("{:<12}".format("角度段"), end="")
    for name in methods:
        print("{:>12}".format(name), end="")
    print()
    print("-" * (12 + 12 * len(methods)))

    for lo, hi in ranges:
        idxs = [i for i, g in enumerate(gts) if lo <= g < hi]
        if not idxs:
            continue
        label = "{}-{}°".format(lo, hi)
        print("{:<12}".format(label), end="")
        for name, preds in methods.items():
            seg_errors = [abs(gts[i] - preds[i]) for i in idxs]
            mae = sum(seg_errors) / len(seg_errors)
            print("{:>12.2f}".format(mae), end="")
        print()

    # 详细结果
    print("\n" + "=" * 50)
    print("详细结果 (前20个样本)")
    print("=" * 50)
    print("{:<20} {:>6}".format("文件", "真实值"), end="")
    for name in methods:
        print("{:>10}".format(name), end="")
    print()
    print("-" * (20 + 6 + 10 * len(methods)))

    for i in range(min(20, len(data))):
        print("{:<20} {:>6.1f}".format(data[i][0], gts[i]), end="")
        for name, preds in methods.items():
            print("{:>10.1f}".format(preds[i]), end="")
        print()


if __name__ == "__main__":
    main()
