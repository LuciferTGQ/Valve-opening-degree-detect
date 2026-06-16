import os
import re
import sys
import csv

sys.path.insert(0, os.path.dirname(__file__))

from src.top.cvnew_predictor import predict_cvnew
from src.common.cnn_predictor import predict_cnn
from src.common.ensemble import predict_ensemble
from src.top.visualize_errors import predict_with_debug, create_visualization

INPUT_DIR = "test_input_top"
MODEL_PATH = "models/mobilenetv3_top.pth"
OUTPUT_DIR = "output"


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def run_prediction(name, predict_fn, images):
    results = []
    for img in images:
        img_path = os.path.join(INPUT_DIR, img)
        angle = predict_fn(img_path)
        results.append((img, angle))
        print(f"  {img}: {angle}°")
    return results


def save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'angle'])
        writer.writerows(results)
    print(f"已保存到 {path}")


def main():
    images = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key=natural_sort_key)
    if not images:
        print(f"{INPUT_DIR}/ 中没有图片")
        return

    print(f"找到 {len(images)} 张图片\n")

    # CVnew
    print("=== CVnew 预测（色块质心 + 几何法 + 弧长比法）===")
    cvnew_results = run_prediction("CVnew", predict_cvnew, images)
    save_csv(cvnew_results, os.path.join(OUTPUT_DIR, "result_cvnew.csv"))

    # CNN
    cnn_results = []
    ens_results = []
    if os.path.exists(MODEL_PATH):
        print("\n=== CNN 预测 ===")
        cnn_results = run_prediction("CNN", lambda p: predict_cnn(p, MODEL_PATH), images)
        save_csv(cnn_results, os.path.join(OUTPUT_DIR, "result_cnn.csv"))

        # 融合
        print("\n=== 融合预测 (CVnew:0.3 + CNN:0.7) ===")
        ens_results = run_prediction("Ensemble", lambda p: predict_ensemble(p, MODEL_PATH), images)
        save_csv(ens_results, os.path.join(OUTPUT_DIR, "result_ensemble.csv"))
    else:
        print(f"\nCNN 模型不存在: {MODEL_PATH}，跳过 CNN 和融合预测")

    # 加载标准答案（如有）
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth_top.csv")
    gt_data = {}
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            for row in csv.DictReader(f):
                gt_data[row['filename']] = float(row['angle'])

    cvnew_dict = dict(cvnew_results)
    cnn_dict = dict(cnn_results)
    ens_dict = dict(ens_results)

    # 对比标准答案
    if gt_data:
        print("\n=== 与标准答案对比 ===")
        print(f"{'文件':<12} {'真实':>6} {'CVnew':>8} {'CNN':>8} {'融合':>8}")
        print("-" * 46)
        for img in images:
            true_angle = gt_data.get(img, '-')
            cvnew_a = cvnew_dict.get(img, '-')
            cnn_a = cnn_dict.get(img, '-')
            ens_a = ens_dict.get(img, '-')
            print(f"{img:<12} {true_angle:>6} {cvnew_a:>8} {cnn_a:>8} {ens_a:>8}")

    # CVnew 调试图输出
    debug_dir = "debug10"
    os.makedirs(debug_dir, exist_ok=True)
    print(f"\n=== CVnew 调试图输出到 {debug_dir}/ ===")
    for i, img in enumerate(images, 1):
        img_path = os.path.join(INPUT_DIR, img)
        pred = cvnew_dict.get(img, 0.0)
        gt_angle = gt_data.get(img, pred)
        debug = predict_with_debug(img_path)
        out_path = os.path.join(debug_dir, f"{i:02d}_{img.replace('.', '_')}.png")
        create_visualization(img, gt_angle, pred, debug, out_path)
        print(f"  {out_path}")

    print(f"已输出 {len(images)} 张调试图到 {debug_dir}/")


if __name__ == "__main__":
    main()
