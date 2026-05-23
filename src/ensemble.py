import os
from cv_predictor import predict_cv
from cnn_predictor import predict_cnn

def predict_ensemble(
    image_path: str,
    model_path: str,
    cv_weight: float = 0.3,
    cnn_weight: float = 0.7
) -> float:
    """
    融合 CV 和 CNN 预测结果

    Args:
        image_path: 图片路径
        model_path: CNN 模型路径
        cv_weight: CV 权重
        cnn_weight: CNN 权重

    Returns:
        融合后的角度预测 (0-80)
    """
    # CV 预测
    try:
        cv_angle = predict_cv(image_path)
    except Exception as e:
        print(f"CV 预测失败: {e}")
        cv_angle = 40.0  # 默认中间值

    # CNN 预测
    if os.path.exists(model_path):
        try:
            cnn_angle = predict_cnn(image_path, model_path)
        except Exception as e:
            print(f"CNN 预测失败: {e}")
            cnn_angle = cv_angle  # 回退到 CV
    else:
        print(f"CNN 模型不存在: {model_path}，使用 CV 预测")
        cnn_angle = cv_angle

    # 加权融合
    final_angle = cv_weight * cv_angle + cnn_weight * cnn_angle

    # 限制范围
    final_angle = min(max(final_angle, 0.0), 80.0)

    return round(final_angle, 1)
