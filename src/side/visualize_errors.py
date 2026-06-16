"""
Side 视角 CV 流程可视化（待 side CV 实现后启用）

结构与 src/top/visualize_errors.py 对齐，预留接口。
"""

# TODO: side CV 预测器实现后，补充完整的可视化逻辑
# 参考 src/top/visualize_errors.py 的 predict_with_debug 和 create_visualization


def predict_with_debug(image_path):
    """运行 side CVnew 流程并返回中间结果（待实现）"""
    raise NotImplementedError("side 视角 CV 可视化尚未实现，请先完成 src/side/cvnew_predictor.py")


def create_visualization(filename, gt_angle, pred_angle, debug, out_path):
    """为单张图片生成调试图（待实现）"""
    raise NotImplementedError("side 视角 CV 可视化尚未实现")
