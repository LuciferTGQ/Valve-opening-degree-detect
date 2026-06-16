import os
from src.common.cnn_predictor import predict_cnn


def _get_cv_predictor(view: str):
    if view == 'side':
        from src.side.cvnew_predictor import predict_cvnew
    else:
        from src.top.cvnew_predictor import predict_cvnew
    return predict_cvnew


def predict_ensemble(
    image_path: str,
    model_path: str,
    cv_weight: float = 0.3,
    cnn_weight: float = 0.7,
    view: str = 'top'
) -> float:
    """
    融合 CVnew 和 CNN 预测结果

    Args:
        image_path: 图片路径
        model_path: CNN 模型路径
        cv_weight: CVnew 权重
        cnn_weight: CNN 权重
        view: 视角 ('top' 或 'side')

    Returns:
        融合后的角度预测 (0-80)
    """
    predict_cvnew = _get_cv_predictor(view)

    # CVnew 预测
    try:
        cv_angle = predict_cvnew(image_path)
    except Exception as e:
        print(f"CVnew 预测失败: {e}")
        cv_angle = 40.0

    # CNN 预测
    if os.path.exists(model_path):
        try:
            cnn_angle = predict_cnn(image_path, model_path)
        except Exception as e:
            print(f"CNN 预测失败: {e}")
            cnn_angle = cv_angle
    else:
        print(f"CNN 模型不存在: {model_path}，使用 CVnew 预测")
        cnn_angle = cv_angle

    # 加权融合
    final_angle = cv_weight * cv_angle + cnn_weight * cnn_angle

    # 限制范围
    final_angle = min(max(final_angle, 0.0), 80.0)

    return round(final_angle, 1)
