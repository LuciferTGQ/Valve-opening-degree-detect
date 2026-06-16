import cv2
import numpy as np
import os
import shutil
from typing import List


def augment_image(image: np.ndarray, seed: int = None) -> np.ndarray:
    """
    对单张图片进行数据增强

    Args:
        image: 原始图片
        seed: 随机种子

    Returns:
        增强后的图片
    """
    if seed is not None:
        np.random.seed(seed)

    img = image.copy()
    h, w = img.shape[:2]

    # 1. 水平翻转
    if np.random.random() > 0.5:
        img = cv2.flip(img, 1)

    # 2. 垂直翻转
    if np.random.random() > 0.7:
        img = cv2.flip(img, 0)

    # 3. 亮度调整
    brightness = np.random.uniform(0.7, 1.3)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)

    # 4. 对比度调整
    contrast = np.random.uniform(0.75, 1.25)
    mean = np.mean(img)
    img = np.clip((img - mean) * contrast + mean, 0, 255).astype(np.uint8)

    # 5. 高斯模糊
    if np.random.random() > 0.5:
        kernel_size = np.random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # 6. 旋转
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    # 7. 高斯噪声
    if np.random.random() > 0.5:
        sigma = np.random.uniform(3, 12)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 8. 缩放（模拟远近拍摄）
    if np.random.random() > 0.5:
        scale = np.random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = cv2.resize(img, (new_w, new_h))
        if scale >= 1.0:
            # 裁剪中心
            dx = (new_w - w) // 2
            dy = (new_h - h) // 2
            img = scaled[dy:dy + h, dx:dx + w]
        else:
            # 填充到原尺寸
            dx = (w - new_w) // 2
            dy = (h - new_h) // 2
            canvas = np.zeros_like(img)
            canvas[dy:dy + new_h, dx:dx + new_w] = scaled
            img = canvas

    # 9. 颜色抖动（随机通道偏移）
    if np.random.random() > 0.5:
        for ch in range(3):
            shift = np.random.uniform(-15, 15)
            img[:, :, ch] = np.clip(img[:, :, ch].astype(float) + shift, 0, 255).astype(np.uint8)

    # 10. 透视变换（模拟不同拍摄角度）
    if np.random.random() > 0.5:
        margin = int(min(h, w) * 0.05)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [np.random.randint(0, margin), np.random.randint(0, margin)],
            [w - np.random.randint(0, margin), np.random.randint(0, margin)],
            [np.random.randint(0, margin), h - np.random.randint(0, margin)],
            [w - np.random.randint(0, margin), h - np.random.randint(0, margin)],
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h))

    return img


def augment_dataset(
    input_dir: str,
    output_dir: str,
    augment_times: int = 95
) -> List[str]:
    """
    对整个数据集进行增强

    Args:
        input_dir: 原始数据目录
        output_dir: 增强后数据目录
        augment_times: 每张图片增强次数

    Returns:
        增强后的文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)

    output_files = []

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue

        # 读取原始图片（支持中文路径）
        img_path = os.path.join(input_dir, filename)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            continue

        # 解析角度 (从文件名)
        name = os.path.splitext(filename)[0]
        angle = float(name.split('_')[1])

        # 复制原始图片（支持中文路径）
        base_name = os.path.splitext(filename)[0]
        original_output = os.path.join(output_dir, f"{base_name}_original.jpg")
        cv2.imencode('.jpg', img)[1].tofile(original_output)
        output_files.append(original_output)

        # 生成增强图片
        for i in range(augment_times):
            augmented = augment_image(img)
            aug_filename = f"{base_name}_aug{i:02d}.jpg"
            aug_path = os.path.join(output_dir, aug_filename)
            cv2.imencode('.jpg', augmented)[1].tofile(aug_path)
            output_files.append(aug_path)

    return output_files


def parse_angle_from_filename(filename: str) -> float:
    """从文件名解析角度"""
    name = os.path.splitext(filename)[0]
    # 处理 original 和 aug 后缀
    if '_original' in name:
        name = name.replace('_original', '')
    elif '_aug' in name:
        name = name.split('_aug')[0]

    parts = name.split('_')
    if len(parts) >= 2:
        return float(parts[1])
    return 0.0
