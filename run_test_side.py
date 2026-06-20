import os
import re
import sys
import csv

sys.path.insert(0, os.path.dirname(__file__))

from src.common.cnn_predictor import predict_cnn
from src.common.ensemble import predict_ensemble
from src.side.cvcnn_predictor import predict_cvcnn

INPUT_DIR = "test_input_side"
NEW_MODEL_PATH = "models/mobilenetv3_side_cv.pth"
OLD_MODEL_PATH = "models/mobilenetv3_side.pth"
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
        print(f"{INPUT_DIR}/ 中没有图片，请先运行:")
        print("  python src/side/prepare_test.py --mode labeled --count 10")
        return

    print(f"找到 {len(images)} 张图片\n")

    # CVnew（预留接口，side CV 未实现时跳过）
    cvnew_results = []
    try:
        from src.side.cvnew_predictor import predict_cvnew
        print("=== CVnew 预测（side 色块质心 + 几何法 + 弧长比法）===")
        cvnew_results = run_prediction("CVnew", predict_cvnew, images)
        save_csv(cvnew_results, os.path.join(OUTPUT_DIR, "result_cvnew_side.csv"))
    except NotImplementedError:
        print("=== CVnew 预测 ===")
        print("side CV 预测器尚未实现，跳过")

    # CNN (新模型: CV裁剪+CNN)
    cvcnn_results = []
    cnn_old_results = []
    ens_results = []
    if os.path.exists(NEW_MODEL_PATH):
        print("\n=== CV+CNN 预测（新模型，CV裁剪+CNN推理）===")
        cvcnn_results = run_prediction("CV+CNN", lambda p: predict_cvcnn(p, NEW_MODEL_PATH), images)
        save_csv(cvcnn_results, os.path.join(OUTPUT_DIR, "result_cvcnn_side.csv"))

        # 旧模型对比
        if os.path.exists(OLD_MODEL_PATH):
            print("\n=== 旧模型对比（直接CNN，无裁剪）===")
            cnn_old_results = run_prediction("CNN-old", lambda p: predict_cnn(p, OLD_MODEL_PATH), images)
            save_csv(cnn_old_results, os.path.join(OUTPUT_DIR, "result_cnn_old_side.csv"))

        # 融合
        if cvnew_results:
            print("\n=== 融合预测 (CVnew:0.3 + CV+CNN:0.7) ===")
            ens_results = run_prediction(
                "Ensemble",
                lambda p: predict_ensemble(p, NEW_MODEL_PATH, view='side'),
                images,
            )
            save_csv(ens_results, os.path.join(OUTPUT_DIR, "result_ensemble_side.csv"))
    else:
        print(f"\nCNN 模型不存在: {NEW_MODEL_PATH}，跳过 CNN 预测")
        print("请先训练 side CV-crop 模型:")
        print("  python src/common/train.py --data 'data_augmented_cropped/side' --model models/mobilenetv3_side_cv.pth --augment 0 --epochs 100")

    # 加载标准答案
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth_side.csv")
    gt_data = {}
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            for row in csv.DictReader(f):
                gt_data[row['filename']] = float(row['angle'])

    cvnew_dict = dict(cvnew_results)
    cvcnn_dict = dict(cvcnn_results)
    cnn_old_dict = dict(cnn_old_results)
    ens_dict = dict(ens_results)

    # 对比标准答案
    if gt_data:
        print("\n=== 与标准答案对比 ===")
        print(f"{'文件':<12} {'真实':>6} {'CVnew':>8} {'CV+CNN':>8} {'CNN旧':>8} {'融合':>8}")
        print("-" * 56)
        for img in images:
            true_angle = gt_data.get(img, '-')
            cvnew_a = cvnew_dict.get(img, '-')
            cvcnn_a = cvcnn_dict.get(img, '-')
            cnn_old_a = cnn_old_dict.get(img, '-')
            ens_a = ens_dict.get(img, '-')
            print(f"{img:<12} {true_angle:>6} {cvnew_a:>8} {cvcnn_a:>8} {cnn_old_a:>8} {ens_a:>8}")

    # CVnew 调试图输出（预留接口）
    try:
        from src.side.visualize_errors import predict_with_debug, create_visualization
        debug_dir = "debug10_side"
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
    except NotImplementedError:
        print("\nside CV 可视化尚未实现，跳过调试图输出")


if __name__ == "__main__":
    main()
