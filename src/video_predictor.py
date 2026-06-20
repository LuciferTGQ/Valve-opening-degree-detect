import argparse
import csv
import inspect
import os
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Callable, Iterable, Optional

import cv2


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


SUPPORTED_ALGORITHMS = {"side_cv", "side_cnn", "top_cv", "top_cnn"}
COMPARISON_MODES = {"side_compare", "top_compare"}
FUSION_MODES = {"side_fusion"}
DEFAULT_MODEL_PATHS = {
    "side_cnn": os.path.join(PROJECT_ROOT, "models", "mobilenetv3_side_cv.pth"),
    "top_cnn": os.path.join(PROJECT_ROOT, "models", "mobilenetv3_top_old.pth"),
}


@dataclass
class FramePrediction:
    """单帧阀门开度预测结果。"""

    frame_index: int
    timestamp_ms: float
    angle: Optional[float]
    algorithm: str
    error: Optional[str] = None


@dataclass
class ComparisonPrediction:
    """同一视频帧的 CV 与 CNN 对比结果。"""

    frame_index: int
    timestamp_ms: float
    cv_angle: Optional[float]
    cnn_angle: Optional[float]
    abs_difference: Optional[float]
    cv_error: Optional[str] = None
    cnn_error: Optional[str] = None


@dataclass
class FusionPrediction:
    """侧面 CNN/CV 门控融合结果。"""

    frame_index: int
    timestamp_ms: float
    cv_angle: Optional[float]
    cnn_angle: Optional[float]
    fusion_angle: Optional[float]
    selected_algorithm: Optional[str]
    cv_error: Optional[str] = None
    cnn_error: Optional[str] = None


@dataclass
class StictionEstimate:
    """整段视频的阀门卡涩程度估计。"""

    level: int
    score: float
    travel_degrees: float
    active_duration_seconds: float
    stall_ratio: float
    jump_ratio: float
    reversal_ratio: float
    sample_count: int


def _write_frame_image(frame, image_path: str) -> None:
    """把视频帧写成临时图片，复用现有按图片路径预测的算法入口。"""
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("failed to encode video frame")
    encoded.tofile(image_path)


def _import_first(candidates: Iterable[tuple[str, str]]) -> Optional[Callable]:
    """按候选模块和函数名查找队友算法入口，便于兼容不同文件命名。"""
    for module_name, function_name in candidates:
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        func = getattr(module, function_name, None)
        if callable(func):
            return func
    return None


def _call_predictor(func: Callable, image_path: str, model_path: Optional[str]) -> float:
    """根据预测函数签名自动决定是否传入 model_path。"""
    signature = inspect.signature(func)
    params = signature.parameters
    if "model_path" in params:
        return float(func(image_path, model_path))
    if len(params) >= 2 and model_path is not None:
        return float(func(image_path, model_path))
    return float(func(image_path))


def _build_full_frame_cnn_predictor(model_path: str) -> Callable[[str], float]:
    """构建整图 CNN 预测器，适用于未使用 CV 裁剪训练的模型。"""
    from PIL import Image
    import torch

    from src.common.cnn_predictor import ValveAngleRegressor, get_transform

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValveAngleRegressor(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    transform = get_transform()

    def predict(image_path: str) -> float:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            angle = float(model(image_tensor).item())
        return round(min(max(angle, 0.0), 80.0), 1)

    return predict


def _build_top_cnn_predictor(model_path: Optional[str]) -> Callable[[str], float]:
    """构建顶视角 CNN 预测器；优先使用队友模块，没有则使用当前项目的 cnn_predictor。"""
    func = _import_first(
        [
            ("top_cnn_predictor", "predict_top_cnn"),
            ("top_cnn_predictor", "predict_cnn"),
            ("top_cnn", "predict_top_cnn"),
            ("top_cnn", "predict_cnn"),
        ]
    )
    if func is not None:
        return lambda image_path: _call_predictor(func, image_path, model_path)

    model_path = model_path or DEFAULT_MODEL_PATHS["top_cnn"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"top_cnn model not found: {model_path}")

    return _build_full_frame_cnn_predictor(model_path)


def _build_side_cnn_predictor(model_path: Optional[str]) -> Callable[[str], float]:
    """构建侧视角 CV 裁剪 + CNN 预测器，视频期间只加载一次模型。"""
    from PIL import Image
    import torch

    from src.common.cnn_predictor import ValveAngleRegressor, get_transform
    from src.side.cvcnn_predictor import _crop_valve_region
    from src.side.cvnew_predictor import (
        _complete_color_mask,
        _enhance_if_dark,
        _filter_green,
        _filter_red,
        _keep_red_adjacent_to_green,
        _read_image,
        _recover_left_red_sliver,
        _recover_nearby_green_fragments,
    )

    resolved_model = model_path or DEFAULT_MODEL_PATHS["side_cnn"]
    if not os.path.exists(resolved_model):
        raise FileNotFoundError(f"side_cnn model not found: {resolved_model}")

    # side_cv 模型使用 CV 裁剪数据训练；旧 side/side_v1 模型使用完整图片训练。
    if "_side_cv" not in os.path.splitext(os.path.basename(resolved_model))[0].lower():
        return _build_full_frame_cnn_predictor(resolved_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValveAngleRegressor(pretrained=False)
    model.load_state_dict(torch.load(resolved_model, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    transform = get_transform()

    def predict(image_path: str) -> float:
        img = _read_image(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        img = _enhance_if_dark(img)
        green_raw = _filter_green(img)
        red_raw = _filter_red(img)
        green_mask = _complete_color_mask(green_raw)
        green_mask = _recover_nearby_green_fragments(green_raw, green_mask)
        red_mask = _complete_color_mask(red_raw)
        red_mask = _recover_left_red_sliver(red_raw, red_mask, green_mask)
        red_mask = _keep_red_adjacent_to_green(green_mask, red_mask)
        cropped = _crop_valve_region(img, green_mask, red_mask)

        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(cropped_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            angle = float(model(tensor).item())
        return round(min(max(angle, 0.0), 80.0), 1)

    return predict


def get_frame_predictor(algorithm: str, model_path: Optional[str] = None) -> Callable[[str], float]:
    """
    根据算法名称返回单帧图片预测函数。

    支持:
    - side_cv: 当前侧视角 OpenCV 算法
    - side_cnn: 队友侧视角 CNN 算法，需提供 side_cnn/side_cnn_predictor 模块
    - top_cv: 当前顶视角 OpenCV/CVnew 算法
    - top_cnn: 队友顶视角 CNN 算法；若未提供模块，则使用当前 cnn_predictor
    """
    normalized = algorithm.lower().strip()
    aliases = {
        "cvnew": "top_cv",
        "top": "top_cv",
        "cnn": "top_cnn",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized == "side_cv":
        from src.side.cvnew_predictor import predict_side_cv

        return lambda image_path: float(predict_side_cv(image_path))
    if normalized == "top_cv":
        from src.top.cvnew_predictor import predict_cvnew

        return lambda image_path: float(predict_cvnew(image_path))
    if normalized == "top_cnn":
        return _build_top_cnn_predictor(model_path)
    if normalized == "side_cnn":
        return _build_side_cnn_predictor(model_path)

    raise ValueError(f"unsupported algorithm: {algorithm}. Supported: {sorted(SUPPORTED_ALGORITHMS)}")


def predict_video(
    video_path: str,
    algorithm: str,
    model_path: Optional[str] = None,
    output_csv: Optional[str] = None,
    output_plot: Optional[str] = None,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    strict: bool = False,
    verbose: bool = False,
) -> list[FramePrediction]:
    """
    对视频逐帧预测阀门开度。

    Args:
        video_path: 输入视频路径。
        algorithm: 识别算法，支持 side_cv、side_cnn、top_cv、top_cnn。
        model_path: CNN 模型路径；CV 算法不需要。
        output_csv: 可选 CSV 输出路径。
        frame_stride: 抽帧间隔，1 表示每帧都识别，2 表示每 2 帧识别一次。
        max_frames: 最多处理多少个被抽取出来的帧。
        strict: 单帧失败时是否直接抛出异常。

    Returns:
        每个被处理帧的预测结果列表。
    """
    if frame_stride <= 0:
        raise ValueError("frame_stride must be greater than 0")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    predictor = get_frame_predictor(algorithm, model_path)
    normalized_algorithm = algorithm.lower().strip()

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video_path}")

    total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    duration_seconds = total_frames / fps if fps > 0 else 0.0
    estimated_count = (total_frames + frame_stride - 1) // frame_stride if total_frames else 0
    if max_frames is not None:
        estimated_count = min(estimated_count, max_frames) if estimated_count else max_frames

    results: list[FramePrediction] = []
    processed_count = 0
    started_at = time.perf_counter()

    if verbose:
        print("=" * 64)
        print("开始分析阀门视频")
        print(f"视频: {os.path.abspath(video_path)}")
        print(f"算法: {normalized_algorithm}")
        print(f"总帧数: {total_frames or '未知'}")
        print(f"帧率: {fps:.3f} FPS" if fps > 0 else "帧率: 未知")
        print(f"时长: {duration_seconds:.3f} 秒" if duration_seconds else "时长: 未知")
        print(f"抽帧间隔: 每 {frame_stride} 帧分析一次")
        print(f"预计分析帧数: {estimated_count or '未知'}")
        print("=" * 64)

    try:
        with tempfile.TemporaryDirectory(prefix="valve_video_frames_") as tmpdir:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % frame_stride != 0:
                    frame_index += 1
                    continue

                timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if timestamp_ms < 0:
                    timestamp_ms = frame_index / fps * 1000.0 if fps > 0 else 0.0
                elif timestamp_ms == 0 and frame_index > 0 and fps > 0:
                    timestamp_ms = frame_index / fps * 1000.0
                frame_path = os.path.join(tmpdir, f"frame_{frame_index:08d}.jpg")
                frame_started_at = time.perf_counter()

                try:
                    _write_frame_image(frame, frame_path)
                    angle = round(float(predictor(frame_path)), 1)
                    result = FramePrediction(frame_index, timestamp_ms, angle, normalized_algorithm)
                except Exception as exc:
                    if strict:
                        raise
                    result = FramePrediction(frame_index, timestamp_ms, None, normalized_algorithm, str(exc))

                results.append(result)
                processed_count += 1
                if verbose:
                    progress = f"{processed_count}/{estimated_count}" if estimated_count else str(processed_count)
                    frame_elapsed = time.perf_counter() - frame_started_at
                    if result.error:
                        print(
                            f"[{progress}] 帧 {frame_index}, 时间 {timestamp_ms / 1000.0:.3f}s, "
                            f"分析失败: {result.error} ({frame_elapsed:.3f}s)"
                        )
                    else:
                        print(
                            f"[{progress}] 帧 {frame_index}, 时间 {timestamp_ms / 1000.0:.3f}s, "
                            f"开度 {result.angle:.1f}° ({frame_elapsed:.3f}s)"
                        )
                frame_index += 1

                if max_frames is not None and processed_count >= max_frames:
                    break
    finally:
        capture.release()

    if output_csv:
        save_video_predictions_csv(results, output_csv)
    if output_plot:
        save_video_predictions_plot(results, output_plot, normalized_algorithm)

    if verbose:
        elapsed = time.perf_counter() - started_at
        success_count = sum(1 for result in results if result.error is None)
        failed_count = len(results) - success_count
        print("=" * 64)
        print("视频分析完成")
        print(f"已分析: {len(results)} 帧，成功: {success_count}，失败: {failed_count}")
        print(f"总耗时: {elapsed:.3f} 秒")
        if output_csv:
            print(f"CSV: {os.path.abspath(output_csv)}")
        if output_plot:
            print(f"曲线图: {os.path.abspath(output_plot)}")
        print("=" * 64)

    return results


def save_video_predictions_csv(results: list[FramePrediction], output_csv: str) -> None:
    """保存逐帧预测结果 CSV。"""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["frame_index", "timestamp_ms", "angle", "algorithm", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def save_video_predictions_plot(
    results: list[FramePrediction],
    output_plot: str,
    algorithm: Optional[str] = None,
) -> None:
    """绘制阀门开度随视频时间变化的曲线图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)
    times = [result.timestamp_ms / 1000.0 for result in results]
    angles = [result.angle if result.angle is not None else float("nan") for result in results]
    valid_times = [result.timestamp_ms / 1000.0 for result in results if result.angle is not None]
    valid_angles = [result.angle for result in results if result.angle is not None]

    figure, axis = plt.subplots(figsize=(10, 5.5))
    if valid_angles:
        axis.plot(times, angles, color="#176B87", linewidth=1.8, label="Opening degree")
        axis.scatter(valid_times, valid_angles, color="#D1495B", s=18, zorder=3, label="Analyzed frame")
        axis.legend(loc="best")
    else:
        axis.text(0.5, 0.5, "No valid predictions", ha="center", va="center", transform=axis.transAxes)

    title_algorithm = algorithm or (results[0].algorithm if results else "unknown")
    axis.set_title(f"Valve Opening Degree Over Time ({title_algorithm})")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Opening degree (degrees)")
    axis.set_ylim(0, 80)
    if valid_times:
        axis.set_xlim(0, max(max(valid_times), 0.001))
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    figure.tight_layout()
    figure.savefig(output_plot, dpi=160)
    plt.close(figure)


def compare_video_algorithms(
    video_path: str,
    view: str,
    model_path: Optional[str] = None,
    output_csv: Optional[str] = None,
    output_plot: Optional[str] = None,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    strict: bool = False,
    verbose: bool = False,
) -> list[ComparisonPrediction]:
    """在同一批视频帧上运行 CV 和 CNN，并输出逐帧对比结果。"""
    normalized_view = view.lower().strip()
    if normalized_view not in {"side", "top"}:
        raise ValueError("view must be 'side' or 'top'")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be greater than 0")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cv_name = f"{normalized_view}_cv"
    cnn_name = f"{normalized_view}_cnn"
    cv_predictor = get_frame_predictor(cv_name)
    cnn_predictor = get_frame_predictor(cnn_name, model_path)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video_path}")

    total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    duration_seconds = total_frames / fps if fps > 0 else 0.0
    estimated_count = (total_frames + frame_stride - 1) // frame_stride if total_frames else 0
    if max_frames is not None:
        estimated_count = min(estimated_count, max_frames) if estimated_count else max_frames

    results: list[ComparisonPrediction] = []
    processed_count = 0
    started_at = time.perf_counter()

    if verbose:
        print("=" * 72)
        print(f"开始进行 {normalized_view} 视角 CV/CNN 对比")
        print(f"视频: {os.path.abspath(video_path)}")
        print(f"总帧数: {total_frames or '未知'}")
        print(f"帧率: {fps:.3f} FPS" if fps > 0 else "帧率: 未知")
        print(f"时长: {duration_seconds:.3f} 秒" if duration_seconds else "时长: 未知")
        print(f"预计分析帧数: {estimated_count or '未知'}")
        print("=" * 72)

    try:
        with tempfile.TemporaryDirectory(prefix="valve_video_compare_") as tmpdir:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_stride != 0:
                    frame_index += 1
                    continue

                timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if timestamp_ms < 0:
                    timestamp_ms = frame_index / fps * 1000.0 if fps > 0 else 0.0
                elif timestamp_ms == 0 and frame_index > 0 and fps > 0:
                    timestamp_ms = frame_index / fps * 1000.0

                frame_path = os.path.join(tmpdir, f"frame_{frame_index:08d}.jpg")
                _write_frame_image(frame, frame_path)
                frame_started_at = time.perf_counter()

                cv_angle = None
                cnn_angle = None
                cv_error = None
                cnn_error = None
                try:
                    cv_angle = round(float(cv_predictor(frame_path)), 1)
                except Exception as exc:
                    cv_error = str(exc)
                    if strict:
                        raise
                try:
                    cnn_angle = round(float(cnn_predictor(frame_path)), 1)
                except Exception as exc:
                    cnn_error = str(exc)
                    if strict:
                        raise

                difference = round(abs(cv_angle - cnn_angle), 1) if cv_angle is not None and cnn_angle is not None else None
                result = ComparisonPrediction(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    cv_angle=cv_angle,
                    cnn_angle=cnn_angle,
                    abs_difference=difference,
                    cv_error=cv_error,
                    cnn_error=cnn_error,
                )
                results.append(result)
                processed_count += 1

                if verbose:
                    progress = f"{processed_count}/{estimated_count}" if estimated_count else str(processed_count)
                    elapsed = time.perf_counter() - frame_started_at
                    cv_text = f"{cv_angle:.1f}°" if cv_angle is not None else f"失败({cv_error})"
                    cnn_text = f"{cnn_angle:.1f}°" if cnn_angle is not None else f"失败({cnn_error})"
                    diff_text = f"{difference:.1f}°" if difference is not None else "无"
                    print(
                        f"[{progress}] 帧 {frame_index}, {timestamp_ms / 1000.0:.3f}s, "
                        f"CV={cv_text}, CNN={cnn_text}, 差值={diff_text} ({elapsed:.3f}s)"
                    )

                frame_index += 1
                if max_frames is not None and processed_count >= max_frames:
                    break
    finally:
        capture.release()

    if output_csv:
        save_video_comparison_csv(results, output_csv)
    if output_plot:
        save_video_comparison_plot(results, output_plot, normalized_view)

    if verbose:
        cv_failed = sum(1 for result in results if result.cv_error)
        cnn_failed = sum(1 for result in results if result.cnn_error)
        print("=" * 72)
        print(f"对比完成，共 {len(results)} 帧，CV 失败 {cv_failed}，CNN 失败 {cnn_failed}")
        print(f"总耗时: {time.perf_counter() - started_at:.3f} 秒")
        if output_csv:
            print(f"CSV: {os.path.abspath(output_csv)}")
        if output_plot:
            print(f"对比图: {os.path.abspath(output_plot)}")
        print("=" * 72)

    return results


def save_video_comparison_csv(results: list[ComparisonPrediction], output_csv: str) -> None:
    """保存 CV/CNN 逐帧对比结果。"""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(ComparisonPrediction.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def save_video_comparison_plot(
    results: list[ComparisonPrediction],
    output_plot: str,
    view: str,
) -> None:
    """在同一张图中用不同颜色绘制 CV 和 CNN 开度曲线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)
    times = [result.timestamp_ms / 1000.0 for result in results]
    cv_angles = [result.cv_angle if result.cv_angle is not None else float("nan") for result in results]
    cnn_angles = [result.cnn_angle if result.cnn_angle is not None else float("nan") for result in results]

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(times, cv_angles, color="#176B87", linewidth=2.0, label="CV")
    axis.plot(times, cnn_angles, color="#D1495B", linewidth=2.0, label="CNN")
    axis.set_title(f"CV vs CNN Valve Opening Degree ({view})")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Opening degree (degrees)")
    axis.set_ylim(0, 80)
    if times:
        axis.set_xlim(0, max(max(times), 0.001))
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_plot, dpi=160)
    plt.close(figure)


def predict_side_fusion(
    video_path: str,
    model_path: Optional[str] = None,
    output_csv: Optional[str] = None,
    output_plot: Optional[str] = None,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    strict: bool = False,
    verbose: bool = False,
    low_threshold: float = 10.0,
    high_threshold: float = 70.0,
    transition_width: float = 5.0,
    cv_probe_interval: int = 15,
    probe_margin: float = 5.0,
) -> list[FusionPrediction]:
    """小/大角度使用侧面 CV，中间角度使用 CNN，边界使用平滑加权。"""
    if not 0.0 <= low_threshold < high_threshold <= 80.0:
        raise ValueError("fusion thresholds must satisfy 0 <= low < high <= 80")
    if transition_width < 0 or transition_width > high_threshold - low_threshold:
        raise ValueError("transition_width must be between 0 and high_threshold - low_threshold")
    if cv_probe_interval < 0:
        raise ValueError("cv_probe_interval must be greater than or equal to 0")
    if probe_margin < 0:
        raise ValueError("probe_margin must be greater than or equal to 0")

    if verbose:
        print("融合规则:")
        print(f"  中心阈值: {low_threshold:.1f}° / {high_threshold:.1f}°")
        print(f"  Smoothstep 过渡带宽: {transition_width:.1f}°")
        print(f"  CNN 区间 CV 探测间隔: {cv_probe_interval} 帧（0 表示关闭周期探测）")
        print(f"  CNN 临界复查余量: {probe_margin:.1f}°")

    if frame_stride <= 0:
        raise ValueError("frame_stride must be greater than 0")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cv_predictor = get_frame_predictor("side_cv")
    cnn_predictor: Optional[Callable[[str], float]] = None
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video_path}")

    total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    estimated_count = (total_frames + frame_stride - 1) // frame_stride if total_frames else 0
    if max_frames is not None:
        estimated_count = min(estimated_count, max_frames) if estimated_count else max_frames

    results: list[FusionPrediction] = []
    processed_count = 0
    started_at = time.perf_counter()
    half_width = transition_width / 2.0
    low_cv_only_end = low_threshold - half_width
    high_cv_only_start = high_threshold + half_width
    low_cnn_probe_limit = low_threshold + half_width + probe_margin
    high_cnn_probe_limit = high_threshold - half_width - probe_margin
    active_mode = "cv"
    frames_since_cv_probe = 0
    cv_run_count = 0
    cnn_run_count = 0

    try:
        with tempfile.TemporaryDirectory(prefix="valve_video_fusion_") as tmpdir:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_stride != 0:
                    frame_index += 1
                    continue

                timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if timestamp_ms < 0:
                    timestamp_ms = frame_index / fps * 1000.0 if fps > 0 else 0.0
                elif timestamp_ms == 0 and frame_index > 0 and fps > 0:
                    timestamp_ms = frame_index / fps * 1000.0

                frame_path = os.path.join(tmpdir, f"frame_{frame_index:08d}.jpg")
                _write_frame_image(frame, frame_path)
                frame_started_at = time.perf_counter()

                cv_angle = None
                cnn_angle = None
                cv_error = None
                cnn_error = None

                # CNN 状态先运行 CNN，只在临界区或达到探测间隔时复查 CV。
                if active_mode == "cnn":
                    try:
                        if cnn_predictor is None:
                            cnn_predictor = get_frame_predictor("side_cnn", model_path)
                        cnn_angle = round(float(cnn_predictor(frame_path)), 1)
                        cnn_run_count += 1
                    except Exception as exc:
                        cnn_error = str(exc)
                        if strict:
                            raise

                    near_boundary = (
                        cnn_angle is not None
                        and (cnn_angle <= low_cnn_probe_limit or cnn_angle >= high_cnn_probe_limit)
                    )
                    periodic_probe = cv_probe_interval > 0 and frames_since_cv_probe >= cv_probe_interval
                    run_cv = cnn_angle is None or near_boundary or periodic_probe
                else:
                    run_cv = True

                if run_cv:
                    try:
                        cv_angle = round(float(cv_predictor(frame_path)), 1)
                        cv_run_count += 1
                    except Exception as exc:
                        cv_error = str(exc)
                        if strict:
                            raise
                    frames_since_cv_probe = 0
                else:
                    frames_since_cv_probe += 1

                # CV 状态确认进入中间区或过渡带后，本帧再按需运行 CNN。
                if active_mode != "cnn":
                    if cv_angle is None:
                        needs_cnn = True
                    elif transition_width <= 0:
                        needs_cnn = low_threshold < cv_angle < high_threshold
                    else:
                        needs_cnn = low_cv_only_end < cv_angle < high_cv_only_start

                    if needs_cnn:
                        try:
                            if cnn_predictor is None:
                                cnn_predictor = get_frame_predictor("side_cnn", model_path)
                            cnn_angle = round(float(cnn_predictor(frame_path)), 1)
                            cnn_run_count += 1
                        except Exception as exc:
                            cnn_error = str(exc)
                            if strict:
                                raise

                selected_algorithm = None
                fusion_angle = None
                if cv_angle is not None and cnn_angle is not None:
                    fusion_angle, selected_algorithm = _smooth_fuse_side_angles(
                        cv_angle,
                        cnn_angle,
                        low_threshold,
                        high_threshold,
                        transition_width,
                    )
                elif cv_angle is not None:
                    selected_algorithm = "side_cv"
                    fusion_angle = cv_angle
                elif cnn_angle is not None:
                    selected_algorithm = "side_cnn" if cv_error is None else "side_cnn_fallback"
                    fusion_angle = cnn_angle

                # 只有 CV 明确确认位于纯中间区后才进入 CNN_ONLY；过渡带保持双算法。
                if cv_angle is not None:
                    in_pure_middle = (
                        low_threshold + half_width <= cv_angle <= high_threshold - half_width
                    )
                    active_mode = "cnn" if in_pure_middle and cnn_angle is not None else "cv"
                elif cnn_angle is not None:
                    active_mode = "cnn"

                result = FusionPrediction(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    cv_angle=cv_angle,
                    cnn_angle=cnn_angle,
                    fusion_angle=fusion_angle,
                    selected_algorithm=selected_algorithm,
                    cv_error=cv_error,
                    cnn_error=cnn_error,
                )
                results.append(result)
                processed_count += 1

                if verbose:
                    progress = f"{processed_count}/{estimated_count}" if estimated_count else str(processed_count)
                    cv_text = f"{cv_angle:.1f}°" if cv_angle is not None else (f"失败({cv_error})" if cv_error else "跳过")
                    cnn_text = f"{cnn_angle:.1f}°" if cnn_angle is not None else "跳过"
                    fusion_text = f"{fusion_angle:.1f}°" if fusion_angle is not None else "失败"
                    elapsed = time.perf_counter() - frame_started_at
                    print(
                        f"[{progress}] 帧 {frame_index}, CV={cv_text}, CNN={cnn_text}, "
                        f"融合={fusion_text}, 来源={selected_algorithm or '无'} ({elapsed:.3f}s)"
                    )

                frame_index += 1
                if max_frames is not None and processed_count >= max_frames:
                    break
    finally:
        capture.release()

    if output_csv:
        save_video_fusion_csv(results, output_csv)
    if output_plot:
        save_video_fusion_plot(
            results, output_plot, low_threshold, high_threshold, transition_width
        )

    if verbose:
        cv_count = sum(1 for result in results if result.selected_algorithm and result.selected_algorithm.startswith("side_cv"))
        cnn_count = sum(1 for result in results if result.selected_algorithm and result.selected_algorithm.startswith("side_cnn"))
        blend_count = sum(1 for result in results if result.selected_algorithm and result.selected_algorithm.startswith("smooth_blend"))
        cnn_skipped = sum(1 for result in results if result.cnn_angle is None and result.cv_angle is not None)
        cv_skipped = sum(1 for result in results if result.cv_angle is None and result.cnn_angle is not None)
        print(
            f"融合完成: 共 {len(results)} 帧，采用 CV {cv_count} 帧，"
            f"采用 CNN {cnn_count} 帧，平滑混合 {blend_count} 帧，"
            f"跳过 CNN {cnn_skipped} 帧，跳过 CV {cv_skipped} 帧，"
            f"CV/CNN 实际调用 {cv_run_count}/{cnn_run_count} 次，"
            f"总耗时 {time.perf_counter() - started_at:.3f}s"
        )
        if output_csv:
            print(f"融合 CSV: {os.path.abspath(output_csv)}")
        if output_plot:
            print(f"融合图: {os.path.abspath(output_plot)}")

    return results


def _smoothstep(value: float) -> float:
    """把 0~1 的线性权重转换成端点斜率为零的平滑权重。"""
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _smooth_fuse_side_angles(
    cv_angle: float,
    cnn_angle: float,
    low_threshold: float,
    high_threshold: float,
    transition_width: float,
) -> tuple[float, str]:
    """根据 CV 角度连续门控，避免 CV/CNN 在阈值处硬切换。"""
    if transition_width <= 0:
        use_cv = cv_angle <= low_threshold or cv_angle >= high_threshold
        return (cv_angle, "side_cv") if use_cv else (cnn_angle, "side_cnn")

    half_width = transition_width / 2.0
    low_start = low_threshold - half_width
    low_end = low_threshold + half_width
    high_start = high_threshold - half_width
    high_end = high_threshold + half_width

    if cv_angle <= low_start:
        return cv_angle, "side_cv"
    if cv_angle < low_end:
        cnn_weight = _smoothstep((cv_angle - low_start) / transition_width)
        angle = (1.0 - cnn_weight) * cv_angle + cnn_weight * cnn_angle
        return round(angle, 1), "smooth_blend_low"
    if cv_angle <= high_start:
        return cnn_angle, "side_cnn"
    if cv_angle < high_end:
        cv_weight = _smoothstep((cv_angle - high_start) / transition_width)
        angle = (1.0 - cv_weight) * cnn_angle + cv_weight * cv_angle
        return round(angle, 1), "smooth_blend_high"
    return cv_angle, "side_cv"


def save_video_fusion_csv(results: list[FusionPrediction], output_csv: str) -> None:
    """保存侧面 CNN/CV 融合结果。"""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(FusionPrediction.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def save_video_fusion_plot(
    results: list[FusionPrediction],
    output_plot: str,
    low_threshold: float,
    high_threshold: float,
    transition_width: float,
) -> None:
    """绘制 CV、CNN 和最终融合开度曲线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)
    times = [result.timestamp_ms / 1000.0 for result in results]
    cv_angles = [result.cv_angle if result.cv_angle is not None else float("nan") for result in results]
    cnn_angles = [result.cnn_angle if result.cnn_angle is not None else float("nan") for result in results]
    fusion_angles = [result.fusion_angle if result.fusion_angle is not None else float("nan") for result in results]

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(times, cv_angles, color="#176B87", linewidth=1.3, alpha=0.55, label="CV")
    axis.plot(times, cnn_angles, color="#D1495B", linewidth=1.3, alpha=0.55, label="CNN")
    axis.plot(times, fusion_angles, color="#2A9D58", linewidth=2.8, label="CNN-CV fusion")
    axis.axhline(low_threshold, color="#777777", linestyle=":", linewidth=1.0, label="Fusion thresholds")
    axis.axhline(high_threshold, color="#777777", linestyle=":", linewidth=1.0)
    if transition_width > 0:
        half_width = transition_width / 2.0
        axis.axhspan(low_threshold - half_width, low_threshold + half_width, color="#777777", alpha=0.08)
        axis.axhspan(high_threshold - half_width, high_threshold + half_width, color="#777777", alpha=0.08)
    axis.set_title("Side CNN-CV Fusion")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Opening degree (degrees)")
    axis.set_ylim(0, 80)
    if times:
        axis.set_xlim(0, max(max(times), 0.001))
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_plot, dpi=160)
    plt.close(figure)


def _prediction_angle(result) -> Optional[float]:
    """从普通预测或融合预测中提取最终开度。"""
    if hasattr(result, "fusion_angle"):
        return result.fusion_angle
    if hasattr(result, "angle"):
        return result.angle
    return None


def _median_smooth(values: list[float], window_size: int = 5) -> list[float]:
    """使用小窗口中值滤波抑制单帧误检尖峰。"""
    if window_size <= 1 or len(values) < 3:
        return values[:]
    radius = window_size // 2
    return [
        float(statistics.median(values[max(0, i - radius) : min(len(values), i + radius + 1)]))
        for i in range(len(values))
    ]


def estimate_stiction(results) -> StictionEstimate:
    """
    根据整段开度时间序列估计三级卡涩程度。

    1级：运行较顺畅；2级：存在一定停滞或突跳；3级：明显停滞-突跳或反向抖动。
    """
    samples = []
    for result in results:
        angle = _prediction_angle(result)
        if angle is None:
            continue
        samples.append((float(result.timestamp_ms) / 1000.0, float(angle)))

    if len(samples) < 5:
        return StictionEstimate(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, len(samples))

    times = [item[0] for item in samples]
    angles = _median_smooth([item[1] for item in samples], 5)
    travel = max(angles) - min(angles)
    total_duration = times[-1] - times[0]
    if travel < 5.0 or total_duration <= 0:
        return StictionEstimate(
            level=1,
            score=0.0,
            travel_degrees=round(travel, 3),
            active_duration_seconds=round(max(total_duration, 0.0), 3),
            stall_ratio=0.0,
            jump_ratio=0.0,
            reversal_ratio=0.0,
            sample_count=len(samples),
        )

    # 去掉开到头或关到头后的正常停留，只分析总行程中间 90% 的运动区间。
    lower_cut = min(angles) + travel * 0.05
    upper_cut = max(angles) - travel * 0.05
    active_indices = [i for i, angle in enumerate(angles) if lower_cut <= angle <= upper_cut]
    if len(active_indices) >= 3:
        start_index = active_indices[0]
        end_index = active_indices[-1]
    else:
        start_index = 0
        end_index = len(angles) - 1

    active_duration = times[end_index] - times[start_index]
    if active_duration <= 0 or end_index - start_index < 2:
        return StictionEstimate(1, 0.0, round(travel, 3), 0.0, 0.0, 0.0, 0.0, len(samples))

    edge_count = max(3, min(10, len(angles) // 10))
    start_level = statistics.median(angles[:edge_count])
    end_level = statistics.median(angles[-edge_count:])
    direction = 1.0 if end_level >= start_level else -1.0
    expected_speed = travel / active_duration
    stall_speed = expected_speed * 0.20
    jump_speed = expected_speed * 2.50

    stall_duration = 0.0
    jump_motion = 0.0
    reverse_motion = 0.0
    total_motion = 0.0
    valid_duration = 0.0

    for index in range(start_index + 1, end_index + 1):
        delta_time = times[index] - times[index - 1]
        if delta_time <= 0:
            continue
        delta_angle = angles[index] - angles[index - 1]
        speed = abs(delta_angle) / delta_time
        motion = abs(delta_angle)
        valid_duration += delta_time
        total_motion += motion

        if speed < stall_speed:
            stall_duration += delta_time
        if speed > jump_speed:
            jump_motion += motion
        if direction * delta_angle < 0:
            reverse_motion += motion

    stall_ratio = stall_duration / valid_duration if valid_duration > 0 else 0.0
    jump_ratio = jump_motion / total_motion if total_motion > 0 else 0.0
    reversal_ratio = reverse_motion / total_motion if total_motion > 0 else 0.0

    # 反向抖动占比通常较小，乘 3 后再限幅以提高区分度。
    score = (
        0.55 * stall_ratio
        + 0.30 * jump_ratio
        + 0.15 * min(1.0, reversal_ratio * 3.0)
    )
    score = min(max(score, 0.0), 1.0)

    if score < 0.12:
        level = 1
    elif score < 0.30:
        level = 2
    else:
        level = 3

    return StictionEstimate(
        level=level,
        score=round(score, 4),
        travel_degrees=round(travel, 3),
        active_duration_seconds=round(active_duration, 3),
        stall_ratio=round(stall_ratio, 4),
        jump_ratio=round(jump_ratio, 4),
        reversal_ratio=round(reversal_ratio, 4),
        sample_count=len(samples),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict valve opening degree for each video frame.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=sorted(SUPPORTED_ALGORITHMS | COMPARISON_MODES | FUSION_MODES),
        help="Recognition algorithm",
    )
    parser.add_argument("--model", default=None, help="CNN model path")
    parser.add_argument("--output", default="output/video_predictions.csv", help="Output CSV path")
    parser.add_argument("--plot", default=None, help="Output plot path; defaults to the CSV name with .png")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every N frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum sampled frames to process")
    parser.add_argument("--strict", action="store_true", help="Raise immediately when a frame fails")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-frame analysis progress")
    parser.add_argument("--fusion-low", type=float, default=10.0, help="CV threshold for the low-angle region")
    parser.add_argument("--fusion-high", type=float, default=70.0, help="CV threshold for the high-angle region")
    parser.add_argument(
        "--fusion-transition",
        type=float,
        default=5.0,
        help="Smoothstep transition width around each fusion threshold",
    )
    parser.add_argument(
        "--fusion-cv-probe-interval",
        type=int,
        default=15,
        help="Run a CV check every N CNN-only frames; 0 disables periodic checks",
    )
    parser.add_argument(
        "--fusion-probe-margin",
        type=float,
        default=5.0,
        help="Force a CV check when CNN approaches a threshold by this margin",
    )
    parser.add_argument(
        "--stiction-test",
        action="store_true",
        help="Estimate and print the three-level valve stiction severity",
    )
    args = parser.parse_args()

    output_plot = args.plot or os.path.splitext(args.output)[0] + ".png"

    if args.algorithm in FUSION_MODES:
        results = predict_side_fusion(
            video_path=args.video,
            model_path=args.model,
            output_csv=args.output,
            output_plot=output_plot,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            strict=args.strict,
            verbose=not args.quiet,
            low_threshold=args.fusion_low,
            high_threshold=args.fusion_high,
            transition_width=args.fusion_transition,
            cv_probe_interval=args.fusion_cv_probe_interval,
            probe_margin=args.fusion_probe_margin,
        )
    elif args.algorithm in COMPARISON_MODES:
        view = args.algorithm.removesuffix("_compare")
        results = compare_video_algorithms(
            video_path=args.video,
            view=view,
            model_path=args.model,
            output_csv=args.output,
            output_plot=output_plot,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            strict=args.strict,
            verbose=not args.quiet,
        )
    else:
        results = predict_video(
            video_path=args.video,
            algorithm=args.algorithm,
            model_path=args.model,
            output_csv=args.output,
            output_plot=output_plot,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            strict=args.strict,
            verbose=not args.quiet,
        )

    if args.stiction_test:
        estimate = estimate_stiction(results)
        print("=" * 64)
        print(f"卡涩程度: {estimate.level} 级")
        print(f"卡涩评分: {estimate.score:.4f}")
        print(f"有效行程: {estimate.travel_degrees:.3f}°")
        print(f"运动时长: {estimate.active_duration_seconds:.3f}s")
        print(f"停滞占比: {estimate.stall_ratio:.4f}")
        print(f"突跳占比: {estimate.jump_ratio:.4f}")
        print(f"反向抖动占比: {estimate.reversal_ratio:.4f}")
        print("=" * 64)

    if args.quiet:
        if args.algorithm in FUSION_MODES:
            failed = sum(1 for result in results if result.fusion_angle is None)
        elif args.algorithm in COMPARISON_MODES:
            failed = sum(1 for result in results if result.cv_error or result.cnn_error)
        else:
            failed = sum(1 for result in results if result.error)
        print(f"processed: {len(results)}")
        print(f"failed: {failed}")
        print(f"saved: {args.output}")
        print(f"plot: {output_plot}")


if __name__ == "__main__":
    main()
