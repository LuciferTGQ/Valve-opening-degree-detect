"""
训练 CNN 模型（支持断点续训、CV裁剪模式）

用法：
  # 训练 top 模型
  python src/common/train.py --data "origin data/top" --model models/mobilenetv3_top.pth --epochs 100 --batch-size 32

  # 训练 side 模型
  python src/common/train.py --data "origin data/side" --model models/mobilenetv3_side.pth --epochs 100 --batch-size 32

  # 训练 side 模型（CV裁剪模式，训练时动态裁剪阀门区域）
  python src/common/train.py --data "origin data/side" --model models/mobilenetv3_side_cv.pth --epochs 100 --crop

  # 暂停后继续（从断点恢复，保留优化器状态和学习率）
  python src/common/train.py --data "origin data/side" --model models/mobilenetv3_side.pth --epochs 100 --resume

  # 从已有模型微调（降低学习率）
  python src/common/train.py --data "origin data/side" --model models/mobilenetv3_side.pth --resume --epochs 50 --lr 0.0003
"""

import os
import sys

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from src.common.cnn_predictor import ValveAngleRegressor, get_transform
from src.common.data_augment import augment_dataset, parse_angle_from_filename

class ValveDataset(Dataset):
    """阀门数据集"""

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]

        # 读取图片
        image = Image.open(img_path).convert('RGB')

        # 解析角度
        filename = os.path.basename(img_path)
        angle = parse_angle_from_filename(filename)

        # 预处理
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([angle], dtype=torch.float32)

def train_model(
    data_dir: str,
    model_save_path: str,
    augment_times: int = 95,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    resume: bool = False,
    crop: bool = False
):
    """
    训练 CNN 模型

    Args:
        data_dir: 原始数据目录
        model_save_path: 模型保存路径
        augment_times: 数据增强倍数
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    print("=" * 50)
    if crop:
        print("开始训练 CNN 模型（CV裁剪模式）")
    else:
        print("开始训练 CNN 模型")
    print("=" * 50)

    # 1. 数据增强（augment_times=0 时跳过，直接使用原图）
    if augment_times == 0:
        augmented_files = [
            os.path.join(data_dir, f)
            for f in sorted(os.listdir(data_dir))
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]
        print(f"跳过增强，直接使用原始数据: {len(augmented_files)} 张")
    else:
        view_name = os.path.basename(os.path.normpath(data_dir))
        augmented_dir = os.path.join(os.path.dirname(data_dir), 'data_augmented', view_name)

        if os.path.isdir(augmented_dir) and len([f for f in os.listdir(augmented_dir) if f.endswith(('.jpg', '.png'))]) > 0:
            augmented_files = [
                os.path.join(augmented_dir, f)
                for f in sorted(os.listdir(augmented_dir))
                if f.endswith(('.jpg', '.png', '.jpeg'))
            ]
            print(f"增强数据已存在，直接使用: {len(augmented_files)} 张")
        else:
            print(f"数据增强中... 增强 {augment_times} 倍")
            augmented_files = augment_dataset(data_dir, augmented_dir, augment_times)
            print(f"增强完成，共 {len(augmented_files)} 张图片")

    # 2. 划分训练集、验证集、测试集 (8:1:1)
    train_files, temp_files = train_test_split(augmented_files, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    print(f"训练集: {len(train_files)} 张，验证集: {len(val_files)} 张，测试集: {len(test_files)} 张")

    # 3. 创建数据集
    transform = get_transform()

    if crop:
        from src.side.crop_dataset import CropValveDataset
        print("CV裁剪模式已启用: 训练时动态裁剪阀门区域")
        train_dataset = CropValveDataset(train_files, transform)
        val_dataset = CropValveDataset(val_files, transform)
        test_dataset = CropValveDataset(test_files, transform)
    else:
        train_dataset = ValveDataset(train_files, transform)
        val_dataset = ValveDataset(val_files, transform)
        test_dataset = ValveDataset(test_files, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 4. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = ValveAngleRegressor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    checkpoint_path = model_save_path.replace('.pth', '_checkpoint.pth')
    best_val_loss = float('inf')
    start_epoch = 0

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 将优化器状态迁移到目标设备（CPU→CUDA 兼容）
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"从 checkpoint 继续: epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    elif resume and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"从模型权重继续（无 optimizer 状态）")
    model.to(device)

    # 6. 训练循环
    for epoch in range(start_epoch, epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for images, angles in train_loader:
            images = images.to(device)
            angles = angles.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, angles)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, angles in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                outputs = model(images)
                loss = criterion(outputs, angles)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 学习率调度
        scheduler.step(val_loss)

        # 保存 checkpoint（每轮都保存，支持断点续训）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'val_loss': val_loss,
        }, checkpoint_path)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} *")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 7. 测试集评估
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device)
            outputs = model(images)
            loss = criterion(outputs, angles)
            test_loss += loss.item()
    test_loss /= len(test_loader)

    print("=" * 50)
    print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"测试集损失: {test_loss:.4f}")
    print(f"模型已保存到: {model_save_path}")
    print("=" * 50)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='训练阀门角度预测模型')
    parser.add_argument('--data', type=str, default='origin data/top', help='原始数据目录')
    parser.add_argument('--model', type=str, default='models/mobilenetv3_top.pth', help='模型保存路径')
    parser.add_argument('--augment', type=int, default=95, help='数据增强倍数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--resume', action='store_true', help='从已有模型继续训练')
    parser.add_argument('--crop', action='store_true', help='启用CV裁剪模式（仅side视角，训练时动态裁剪阀门区域）')

    args = parser.parse_args()

    # 确保模型目录存在
    os.makedirs(os.path.dirname(args.model), exist_ok=True)

    train_model(
        data_dir=args.data,
        model_save_path=args.model,
        augment_times=args.augment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume=args.resume,
        crop=args.crop
    )
