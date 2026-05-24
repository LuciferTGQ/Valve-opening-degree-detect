import os
import sys
import csv
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cvnew_predictor import predict_cvnew
from cnn_predictor import predict_cnn

DATA_DIR = "origin data/top"
MODEL_PATH = "models/mobilenetv3_top.pth"
OUTPUT_DIR = "output/eval"


def load_ground_truth(data_dir):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))])
    gt = []
    for f in files:
        angle = float(f.split('_')[1].replace('.jpg', '').replace('.png', ''))
        gt.append((f, angle))
    return gt


def calc_metrics(gts, preds):
    errors = [abs(g - p) for g, p in zip(gts, preds)]
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
    max_err = max(errors)
    return mae, rmse, max_err


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_ground_truth(DATA_DIR)
    filenames = [d[0] for d in data]
    gts = [d[1] for d in data]
    paths = [os.path.join(DATA_DIR, f) for f in filenames]

    print(f"加载 {len(data)} 张图片\n")

    print("正在运行 CVnew 预测...")
    cvnew_preds = [predict_cvnew(p) for p in paths]

    cnn_preds = None
    if os.path.exists(MODEL_PATH):
        print("正在运行 CNN 预测...")
        cnn_preds = [predict_cnn(p, MODEL_PATH) for p in paths]
    else:
        print(f"CNN 模型不存在: {MODEL_PATH}，跳过")

    methods = {'CVnew': cvnew_preds}
    if cnn_preds:
        methods['CNN'] = cnn_preds

    print("\n" + "=" * 50)
    print("整体指标")
    print("=" * 50)
    print("{:<10} {:>6} {:>7} {:>8}".format("方法", "MAE", "RMSE", "最大误差"))
    print("-" * 34)
    for name, preds in methods.items():
        mae, rmse, max_err = calc_metrics(gts, preds)
        print("{:<10} {:>6.1f} {:>7.1f} {:>8.1f}".format(name, mae, rmse, max_err))

    print("\n" + "=" * 50)
    print("分段指标 (MAE)")
    print("=" * 50)
    ranges = [(0, 20), (20, 40), (40, 60), (60, 80)]
    print("{:<12}".format("角度段"), end="")
    for name in methods:
        print("{:>10}".format(name), end="")
    print()
    print("-" * (12 + 10 * len(methods)))

    for lo, hi in ranges:
        idxs = [i for i, g in enumerate(gts) if lo <= g < hi]
        if not idxs:
            continue
        label = "{}-{}°".format(lo, hi)
        print("{:<12}".format(label), end="")
        for name, preds in methods.items():
            seg_errors = [abs(gts[i] - preds[i]) for i in idxs]
            mae = sum(seg_errors) / len(seg_errors)
            print("{:>10.1f}".format(mae), end="")
        print()

    csv_path = os.path.join(OUTPUT_DIR, "detailed_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['filename', 'ground_truth', 'CVnew']
        if cnn_preds:
            header.append('CNN')
        writer.writerow(header)
        for i in range(len(filenames)):
            row = [filenames[i], gts[i], cvnew_preds[i]]
            if cnn_preds:
                row.append(cnn_preds[i])
            writer.writerow(row)
    print(f"\n详细结果已保存到 {csv_path}")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0][0]
    x = np.arange(len(gts))
    ax.plot(x, gts, 'k-', label='真实值', linewidth=2)
    ax.plot(x, cvnew_preds, 'g--', label='CVnew', alpha=0.8)
    if cnn_preds:
        ax.plot(x, cnn_preds, 'r--', label='CNN', alpha=0.8)
    ax.set_xlabel('样本序号')
    ax.set_ylabel('角度 (°)')
    ax.set_title('预测值 vs 真实值')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0][1]
    cvnew_errors = [g - p for g, p in zip(gts, cvnew_preds)]
    ax.scatter(gts, cvnew_errors, c='green', alpha=0.6, label='CVnew', s=30)
    if cnn_preds:
        cnn_errors = [g - p for g, p in zip(gts, cnn_preds)]
        ax.scatter(gts, cnn_errors, c='red', alpha=0.6, label='CNN', s=30)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('真实角度 (°)')
    ax.set_ylabel('误差 (真实-预测) (°)')
    ax.set_title('误差分布 (正值=低估, 负值=高估)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    cvnew_abs_err = [abs(g - p) for g, p in zip(gts, cvnew_preds)]
    ax.bar(x, cvnew_abs_err, 0.4, label='CVnew', color='green', alpha=0.7)
    if cnn_preds:
        cnn_abs_err = [abs(g - p) for g, p in zip(gts, cnn_preds)]
        ax.bar(x + 0.4, cnn_abs_err, 0.4, label='CNN', color='red', alpha=0.7)
    ax.set_xlabel('样本序号')
    ax.set_ylabel('绝对误差 (°)')
    ax.set_title('各样本绝对误差')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1][1]
    range_labels = []
    mae_data = {name: [] for name in methods}
    for lo, hi in ranges:
        idxs = [i for i, g in enumerate(gts) if lo <= g < hi]
        if not idxs:
            continue
        range_labels.append("{}-{}°".format(lo, hi))
        for name, preds in methods.items():
            seg_err = [abs(gts[i] - preds[i]) for i in idxs]
            mae_data[name].append(sum(seg_err) / len(seg_err))

    x = np.arange(len(range_labels))
    n = len(methods)
    width = 0.8 / n
    colors = {'CVnew': 'green', 'CNN': 'red'}
    for i, (name, vals) in enumerate(mae_data.items()):
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=name, color=colors[name], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels)
    ax.set_xlabel('角度范围')
    ax.set_ylabel('MAE (°)')
    ax.set_title('分段 MAE 对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.savefig(fig_path, dpi=150)
    print(f"对比图已保存到 {fig_path}")


if __name__ == "__main__":
    main()
