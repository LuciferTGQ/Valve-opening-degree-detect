# 阀门开度识别系统设计文档

## 目标

输入：文件夹路径，内含 `1.png` 到 `N.png`
输出：CSV文件，两列 `filename,angle`，角度精度0.1°

## 架构

```
输入文件夹 → 逐张读取 → CV预测 + CNN预测 → 加权融合 → 输出CSV
```

## 模块设计

### 1. CV Predictor（cv_predictor.py）

**算法**：
1. 读取图片 → 灰度化 → 高斯模糊
2. HoughCircles 检测圆形阀门
3. 创建圆形掩膜
4. 转HSV，提取绿色区域
5. 形态学开运算去噪
6. 计算绿色占比 → 角度 = green_ratio * 80.0

**光线鲁棒性**：HSV色彩空间 + 形态学去噪

### 2. CNN Predictor（cnn_predictor.py）

**模型**：MobileNetV3-Small（预训练ImageNet）

**数据增强**（10倍扩充）：
- 水平/垂直翻转
- 亮度/对比度调整（±30%）
- 高斯模糊
- 旋转（±15°）
- 随机裁剪

**训练配置**：
- 输入：224×224 RGB
- 输出：角度值（0-80）
- 损失：MSELoss
- 优化器：Adam, lr=1e-3
- epochs：100

### 3. Ensemble（ensemble.py）

```python
final_angle = 0.3 * cv_angle + 0.7 * cnn_angle
```

### 4. 主入口（predict.py）

```bash
python src/predict.py --input "path/to/folder" --output "output/result.csv"
```

## 文件结构

```
├── src/
│   ├── cv_predictor.py
│   ├── cnn_predictor.py
│   ├── ensemble.py
│   ├── data_augment.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── mobilenetv3_top.pth
├── data_augmented/
│   └── top/
└── output/
    └── result.csv
```

## 成功标准

- 能跑通完整流程
- CNN精度优于CV（目标±5°以内）
- 输出格式正确
