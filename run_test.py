import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cv_predictor import predict_cv
from cvnew_predictor import predict_cvnew
from cnn_predictor import predict_cnn
from ensemble import predict_ensemble

INPUT_DIR = "test_input"
MODEL_PATH = "models/mobilenetv3_top.pth"
OUTPUT_DIR = "output"


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
    images = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        print(f"test_input/ 中没有图片")
        return

    print(f"找到 {len(images)} 张图片\n")

    # CV
    print("=== CV 预测 ===")
    cv_results = run_prediction("CV", predict_cv, images)
    save_csv(cv_results, os.path.join(OUTPUT_DIR, "result_cv.csv"))

    # CVnew
    print("\n=== CVnew 预测（内圈法）===")
    cvnew_results = run_prediction("CVnew", predict_cvnew, images)
    save_csv(cvnew_results, os.path.join(OUTPUT_DIR, "result_cvnew.csv"))

    # CNN
    if os.path.exists(MODEL_PATH):
        print("\n=== CNN 预测 ===")
        cnn_results = run_prediction("CNN", lambda p: predict_cnn(p, MODEL_PATH), images)
        save_csv(cnn_results, os.path.join(OUTPUT_DIR, "result_cnn.csv"))

        # 融合
        print("\n=== 融合预测 (CV:0.3 + CNN:0.7) ===")
        ens_results = run_prediction("Ensemble", lambda p: predict_ensemble(p, MODEL_PATH), images)
        save_csv(ens_results, os.path.join(OUTPUT_DIR, "result_ensemble.csv"))
    else:
        print(f"\nCNN 模型不存在: {MODEL_PATH}，跳过 CNN 和融合预测")

    # 对比标准答案
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth.csv")
    if os.path.exists(gt_path):
        print("\n=== 与标准答案对比 ===")
        gt = {}
        with open(gt_path) as f:
            for row in csv.DictReader(f):
                gt[row['filename']] = float(row['angle'])

        print(f"{'文件':<12} {'真实':>6} {'CV':>8} {'CVnew':>8} {'CNN':>8} {'融合':>8}")
        print("-" * 56)
        cv_dict = dict(cv_results)
        cvnew_dict = dict(cvnew_results)
        cnn_dict = dict(cnn_results) if os.path.exists(MODEL_PATH) else {}
        ens_dict = dict(ens_results) if os.path.exists(MODEL_PATH) else {}

        for img in images:
            true_angle = gt.get(img, '-')
            cv_a = cv_dict.get(img, '-')
            cvnew_a = cvnew_dict.get(img, '-')
            cnn_a = cnn_dict.get(img, '-')
            ens_a = ens_dict.get(img, '-')
            print(f"{img:<12} {true_angle:>6} {cv_a:>8} {cvnew_a:>8} {cnn_a:>8} {ens_a:>8}")


if __name__ == "__main__":
    main()
