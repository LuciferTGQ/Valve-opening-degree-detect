import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from cnn_predictor import ValveAngleRegressor, get_transform
from data_augment import augment_dataset, parse_angle_from_filename

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
    augment_times: int = 10,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3
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
    print("开始训练 CNN 模型")
    print("=" * 50)

    # 1. 数据增强
    augmented_dir = os.path.join(os.path.dirname(data_dir), 'data_augmented', 'top')
    print(f"数据增强中... 增强 {augment_times} 倍")
    augmented_files = augment_dataset(data_dir, augmented_dir, augment_times)
    print(f"增强完成，共 {len(augmented_files)} 张图片")

    # 2. 划分训练集和验证集
    train_files, val_files = train_test_split(augmented_files, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_files)} 张，验证集: {len(val_files)} 张")

    # 3. 创建数据集
    transform = get_transform()
    train_dataset = ValveDataset(train_files, transform)
    val_dataset = ValveDataset(val_files, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = ValveAngleRegressor()
    model.to(device)

    # 5. 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # 6. 训练循环
    best_val_loss = float('inf')

    for epoch in range(epochs):
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

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} *")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("=" * 50)
    print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存到: {model_save_path}")
    print("=" * 50)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='训练阀门角度预测模型')
    parser.add_argument('--data', type=str, default='origin data/top', help='原始数据目录')
    parser.add_argument('--model', type=str, default='models/mobilenetv3_top.pth', help='模型保存路径')
    parser.add_argument('--augment', type=int, default=10, help='数据增强倍数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')

    args = parser.parse_args()

    # 确保模型目录存在
    os.makedirs(os.path.dirname(args.model), exist_ok=True)

    train_model(
        data_dir=args.data,
        model_save_path=args.model,
        augment_times=args.augment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
