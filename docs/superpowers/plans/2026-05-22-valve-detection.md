# 阀门开度识别系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 输入阀门图片文件夹，输出CSV角度预测结果

**Architecture:** CV（HoughCircles + 绿色占比）+ CNN（MobileNetV3-Small）加权融合

**Tech Stack:** Python 3.x, OpenCV, PyTorch, torchvision, pandas

---

## 文件结构

```
G:\大学资料\南航\iot课设\
├── src/
│   ├── cv_predictor.py         # CV预测模块
│   ├── cnn_predictor.py        # CNN预测模块
│   ├── ensemble.py             # 融合逻辑
│   ├── data_augment.py         # 数据增强
│   ├── train.py                # CNN训练脚本
│   └── predict.py              # 主入口
├── tests/
│   ├── test_cv_predictor.py    # CV模块测试
│   ├── test_cnn_predictor.py   # CNN模块测试
│   └── test_predict.py         # 主入口测试
├── models/                     # 训练好的模型
├── data_augmented/top/         # 增强后的数据
└── output/                     # 输出结果
```

---

## Task 1: 环境配置与依赖安装

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: 创建 requirements.txt**

```txt
opencv-python>=4.7.0
torch>=2.1.0
torchvision>=0.16.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

- [ ] **Step 2: 安装依赖**

Run: `pip install -r requirements.txt`
Expected: 成功安装所有依赖

- [ ] **Step 3: 验证安装**

Run: `python -c "import cv2, torch, torchvision, pandas; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requirements.txt"
```

---

## Task 2: CV Predictor 模块

**Files:**
- Create: `src/cv_predictor.py`
- Create: `tests/test_cv_predictor.py`

- [ ] **Step 1: 编写 CV 预测函数**

```python
# src/cv_predictor.py
import cv2
import numpy as np

def predict_cv(image_path: str) -> float:
    """
    使用 OpenCV 预测阀门开度角度
    
    Args:
        image_path: 图片路径
        
    Returns:
        预测角度 (0-80)
    """
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 2. 灰度化 + 高斯模糊
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 3. HoughCircles 检测圆形
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=50,
        param1=100, 
        param2=30, 
        minRadius=20, 
        maxRadius=200
    )
    
    # 4. 创建掩膜
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 选择最大圆
        max_circle = max(circles[0], key=lambda c: c[2])
        cv2.circle(mask, (max_circle[0], max_circle[1]), max_circle[2], 255, -1)
    
    # 5. 转 HSV，提取绿色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 绿色范围 (H: 35-85)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 6. 应用圆形掩膜
    green_mask = cv2.bitwise_and(green_mask, mask)
    
    # 7. 形态学开运算去噪
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    
    # 8. 计算绿色占比
    total_pixels = cv2.countNonZero(mask)
    green_pixels = cv2.countNonZero(green_mask)
    
    if total_pixels == 0:
        return 0.0
    
    green_ratio = green_pixels / total_pixels
    angle = green_ratio * 80.0
    
    return round(min(max(angle, 0.0), 80.0), 1)
```

- [ ] **Step 2: 编写测试**

```python
# tests/test_cv_predictor.py
import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cv_predictor import predict_cv

def test_predict_cv_with_valid_image():
    """测试有效图片的预测"""
    # 使用 origin data/top/ 中的第一张图片
    image_path = "origin data/top/0001_3.4.jpg"
    if os.path.exists(image_path):
        angle = predict_cv(image_path)
        assert 0 <= angle <= 80
        assert isinstance(angle, float)

def test_predict_cv_invalid_path():
    """测试无效路径"""
    with pytest.raises(ValueError):
        predict_cv("nonexistent.jpg")
```

- [ ] **Step 3: 运行测试**

Run: `cd "G:/大学资料/南航/iot课设" && python -m pytest tests/test_cv_predictor.py -v`
Expected: 测试通过

- [ ] **Step 4: Commit**

```bash
git add src/cv_predictor.py tests/test_cv_predictor.py
git commit -m "feat: add CV predictor with HoughCircles"
```

---

## Task 3: 数据增强模块

**Files:**
- Create: `src/data_augment.py`

- [ ] **Step 1: 编写数据增强函数**

```python
# src/data_augment.py
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
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    
    return img

def augment_dataset(
    input_dir: str, 
    output_dir: str, 
    augment_times: int = 10
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
    
    for filename in os.listdir(input_dir):
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        # 读取原始图片
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # 解析角度 (从文件名)
        name = os.path.splitext(filename)[0]
        angle = float(name.split('_')[1])
        
        # 复制原始图片
        base_name = os.path.splitext(filename)[0]
        original_output = os.path.join(output_dir, f"{base_name}_original.jpg")
        cv2.imwrite(original_output, img)
        output_files.append(original_output)
        
        # 生成增强图片
        for i in range(augment_times):
            augmented = augment_image(img, seed=i)
            aug_filename = f"{base_name}_aug{i:02d}.jpg"
            aug_path = os.path.join(output_dir, aug_filename)
            cv2.imwrite(aug_path, augmented)
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
```

- [ ] **Step 2: Commit**

```bash
git add src/data_augment.py
git commit -m "feat: add data augmentation module"
```

---

## Task 4: CNN Predictor 模块

**Files:**
- Create: `src/cnn_predictor.py`

- [ ] **Step 1: 编写 CNN 预测函数**

```python
# src/cnn_predictor.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class ValveAngleRegressor(nn.Module):
    """阀门角度回归模型"""
    
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        # 替换最后的分类层
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_transform():
    """获取图片预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict_cnn(image_path: str, model_path: str) -> float:
    """
    使用 CNN 预测阀门开度角度
    
    Args:
        image_path: 图片路径
        model_path: 模型路径
        
    Returns:
        预测角度 (0-80)
    """
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ValveAngleRegressor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. 加载图片
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 3. 预测
    with torch.no_grad():
        output = model(image_tensor)
        angle = output.item()
    
    # 4. 限制范围
    angle = min(max(angle, 0.0), 80.0)
    
    return round(angle, 1)
```

- [ ] **Step 2: Commit**

```bash
git add src/cnn_predictor.py
git commit -m "feat: add CNN predictor module"
```

---

## Task 5: CNN 训练脚本

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: 编写训练脚本**

```python
# src/train.py
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
```

- [ ] **Step 2: Commit**

```bash
git add src/train.py
git commit -m "feat: add CNN training script"
```

---

## Task 6: Ensemble 融合模块

**Files:**
- Create: `src/ensemble.py`

- [ ] **Step 1: 编写融合函数**

```python
# src/ensemble.py
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
```

- [ ] **Step 2: Commit**

```bash
git add src/ensemble.py
git commit -m "feat: add ensemble module"
```

---

## Task 7: 主入口预测脚本

**Files:**
- Create: `src/predict.py`

- [ ] **Step 1: 编写主入口**

```python
# src/predict.py
import os
import sys
import argparse
import pandas as pd
from typing import List, Tuple

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ensemble import predict_ensemble
from cv_predictor import predict_cv
from cnn_predictor import predict_cnn

def predict_folder(
    folder_path: str,
    output_csv: str,
    model_path: str = None,
    use_cnn: bool = True
) -> List[Tuple[str, float]]:
    """
    预测文件夹中所有图片的角度
    
    Args:
        folder_path: 图片文件夹路径
        output_csv: 输出 CSV 路径
        model_path: CNN 模型路径
        use_cnn: 是否使用 CNN
        
    Returns:
        [(filename, angle), ...]
    """
    # 1. 获取所有图片
    image_files = []
    for f in sorted(os.listdir(folder_path)):
        if f.endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(f)
    
    if not image_files:
        print(f"文件夹中没有图片: {folder_path}")
        return []
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 2. 逐张预测
    results = []
    
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        
        if use_cnn and model_path and os.path.exists(model_path):
            # 使用融合预测
            angle = predict_ensemble(image_path, model_path)
            method = "ensemble"
        else:
            # 仅使用 CV
            angle = predict_cv(image_path)
            method = "cv"
        
        results.append((filename, angle))
        print(f"{filename}: {angle}° ({method})")
    
    # 3. 保存 CSV
    df = pd.DataFrame(results, columns=['filename', 'angle'])
    df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到: {output_csv}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='预测阀门开度角度')
    parser.add_argument('--input', type=str, required=True, help='输入图片文件夹')
    parser.add_argument('--output', type=str, default='output/result.csv', help='输出 CSV 路径')
    parser.add_argument('--model', type=str, default='models/mobilenetv3_top.pth', help='CNN 模型路径')
    parser.add_argument('--no-cnn', action='store_true', help='不使用 CNN，仅用 CV')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    predict_folder(
        folder_path=args.input,
        output_csv=args.output,
        model_path=args.model,
        use_cnn=not args.no_cnn
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/predict.py
git commit -m "feat: add main prediction script"
```

---

## Task 8: 端到端测试

**Files:**
- Create: `tests/test_predict.py`

- [ ] **Step 1: 编写端到端测试**

```python
# tests/test_predict.py
import pytest
import os
import sys
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import predict_folder

def test_predict_folder_cv_only():
    """测试仅使用 CV 的预测"""
    # 使用 origin data/top 中的前 10 张图片
    input_dir = "origin data/top"
    
    if not os.path.exists(input_dir):
        pytest.skip("测试数据不存在")
    
    # 创建临时输出目录
    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = os.path.join(tmpdir, "result.csv")
        
        # 获取前 10 张图片
        images = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])[:10]
        
        # 创建测试文件夹
        test_dir = os.path.join(tmpdir, "test_images")
        os.makedirs(test_dir)
        for img in images:
            shutil.copy(os.path.join(input_dir, img), test_dir)
        
        # 运行预测
        results = predict_folder(test_dir, output_csv, use_cnn=False)
        
        # 验证
        assert len(results) == 10
        for filename, angle in results:
            assert 0 <= angle <= 80
            assert isinstance(angle, float)
        
        # 验证 CSV 文件
        assert os.path.exists(output_csv)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: 运行测试**

Run: `cd "G:/大学资料/南航/iot课设" && python -m pytest tests/test_predict.py -v`
Expected: 测试通过

- [ ] **Step 3: Commit**

```bash
git add tests/test_predict.py
git commit -m "test: add end-to-end prediction test"
```

---

## Task 9: 完整流程演示

- [ ] **Step 1: 训练 CNN 模型**

Run: `cd "G:/大学资料/南航/iot课设" && python src/train.py --data "origin data/top" --model "models/mobilenetv3_top.pth" --augment 10 --epochs 50`
Expected: 训练完成，模型保存到 models/mobilenetv3_top.pth

- [ ] **Step 2: 使用 CV 预测测试**

Run: `cd "G:/大学资料/南航/iot课设" && python src/predict.py --input "origin data/top" --output "output/result_cv.csv" --no-cnn`
Expected: 生成 result_cv.csv

- [ ] **Step 3: 使用融合预测测试**

Run: `cd "G:/大学资料/南航/iot课设" && python src/predict.py --input "origin data/top" --output "output/result_ensemble.csv"`
Expected: 生成 result_ensemble.csv

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: complete valve detection pipeline"
```

---

## 执行选项

Plan complete and saved to `docs/superpowers/plans/2026-05-22-valve-detection.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
