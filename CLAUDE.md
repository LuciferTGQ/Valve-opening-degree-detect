# 阀门开度识别系统 (Valve Opening Degree Detection)

## 项目概述

安特威阀门有限公司 × 南京航空航天大学 IoT 课设项目。通过计算机视觉技术，从阀门图像中识别开度指示器角度（0-80°，精度 0.1°），进而判断阀门卡涩程度。

卡涩程度判定函数由安特威提供（黑盒），我们只需**输入图片 → 输出开度角度**，函数自行判断卡涩等级。

## 系统架构

```
移动端 App → 拍摄视频 → 上传服务器 → 抽帧 → CV/CNN推理 → 角度序列 → 卡涩判定 → 返回结果
```

- **前端**：移动端 App，支持拍摄视频、上传、查看结果
- **后端**：Python 服务器，负责视频处理和 AI 推理
- **算法核心**：CV 弧长比法 + CNN 回归模型融合，输出开度角度（0.0-80.0）

## 技术方案

本项目采用 CV + CNN 双通道融合：

1. **CV 方法（cvnew_predictor）**：
   - **top 视角**：色块质心定位 → 几何法精确定位圆心 → 色块配对优化 → 多半径弧长比法计算角度
   - **side 视角**：红绿颜色带投影宽度 → 多扫描线测宽 → 圆弦几何关系换算角度（队友实现）
2. **CNN 方法（cnn_predictor）**：MobileNetV3-Small 回归模型，输入单帧图像 → 输出角度
3. **CV+CNN 方法（cvcnn_predictor）**：CV 提取红绿色块区域 → 裁剪阀门区域 → 喂给 CNN 推理（仅 side 视角）
4. **融合（ensemble）**：CV:0.3 + CNN:0.7 加权融合

### 数据增强策略

- 水平翻转（50%）、垂直翻转（30%）
- 亮度调节（0.7-1.3x）、对比度调节（0.75-1.25x）
- 高斯模糊（kernel 3/5/7，50%）
- 旋转（-15° ~ +15°）
- 高斯噪声（σ 3-12，50%）
- 缩放（0.8-1.2x，50%，模拟远近拍摄）
- 颜色抖动（通道偏移 ±15，50%）
- 透视变换（5% 边距，50%，模拟拍摄角度偏差）
- 每张原图增强 95 次（默认）

## 数据结构

### 目录布局

```
origin data/
├── top/                    # 顶部视角原图（jpg）
├── side/                   # 正侧面视角原图（jpg + png 混合）
├── data_augmented/
│   ├── top/                # 增强后的顶部图像
│   └── side/               # 增强后的侧面图像
├── video/                  # 原始视频文件
│   ├── *.mp4               # 顶部和侧面拍摄视频
│   ├── side1/              # 侧面1抽帧结果
│   └── side2/              # 侧面2抽帧结果
└── 蒙板/                   # 拍摄蒙版模板

新增阀门数据集6.3/          # 新增数据（220张/视角，含原图+增强混合，角度 6.7-46.6°）
├── top/
└── side/

阀门测试集6.2/              # 测试集（20张/视角，CSV 记录标准答案）
├── top/
│   └── top.csv
└── side/
    └── side.csv

test_input_top/             # top 测试图片（20张，来自 6.2）
test_input_side/            # side 测试图片（20张，来自 6.2）
finetune_data/top/          # 6.3 top 数据（用于微调实验）
output/                     # 预测结果 CSV + ground truth
```

### 文件命名规范

- `origin data/top/` 和 `origin data/side/`：`NNNN_angle.jpg`，序号 4 位零填充，角度精确到 1 位小数
- `新增阀门数据集6.3/`：`NNNN_angle.jpg`，序号从 0159 起
- `阀门测试集6.2/`：`N.jpg`（序号无零填充），角度在 CSV 文件中
- 角度范围：0.0-80.0 度（全绿=80°，全红=0°）

## 阀门类型

本项目**只关注 top 和 side 两种视角**，只关注**红绿阀门**。

## 核心约束

- **输出精度**：0.1 度
- **角度范围**：0-80 度
- **全绿 = 80°，全红 = 0°**
- 只考虑红绿阀门，且只考虑正上方和正侧方，训练两个独立模型
- 数据增强是必需的（原始数据量不足以泛化真实场景）
- CV 方法在极低角度（<5°）时绿色饱和度过低，难以检测（固有盲区）

## 代码结构

```
run_test_top.py              # top 视角快速测试入口（CVnew + CNN + 融合 + 调试图）
run_test_side.py             # side 视角快速测试入口（CVnew + CNN + CV+CNN + 融合）
test_input_top/              # top 测试图片（20张）
test_input_side/             # side 测试图片（20张）
src/
├── common/                  # 共用代码（视角无关）
│   ├── cnn_predictor.py     # CNN 模型定义（MobileNetV3-Small）+ 推理
│   ├── ensemble.py          # CV + CNN 加权融合（默认 CV:0.3, CNN:0.7）
│   ├── data_augment.py      # 数据增强（翻转、亮度、对比度、模糊、旋转、噪声、缩放等）
│   ├── train.py             # 训练循环（支持断点续训、--augment 0 跳过增强）
│   ├── evaluate.py          # 评估指标 + matplotlib 可视化
│   └── predict.py           # 批量预测（文件夹 → CSV）
├── top/                     # top 视角专用
│   ├── cvnew_predictor.py   # CV 预测器（色块质心 + 几何法 + 弧长比法）
│   ├── visualize_errors.py  # CV 流程可视化（2x3 拼图）
│   └── prepare_test.py      # 测试数据准备
└── side/                    # side 视角专用
    ├── cvnew_predictor.py   # CV 预测器（红绿颜色带投影宽度 + 圆弦几何法）
    ├── cvcnn_predictor.py   # CV+CNN 预测器（CV 裁剪阀门区域 → CNN 推理）
    ├── evaluate.py          # side 评估脚本
    └── prepare_test.py      # 测试数据准备
tests/
├── test_predict_cvnew_top.py  # CV 预测测试
├── test_predict_cnn.py        # CNN 预测测试
└── test_predict_ensemble.py   # 融合预测测试
models/
├── mobilenetv3_top.pth        # 已训练的顶部视角模型
├── mobilenetv3_top_checkpoint.pth   # top 训练 checkpoint
├── mobilenetv3_side.pth       # 已训练的侧面视角模型
└── mobilenetv3_side_checkpoint.pth  # side 训练 checkpoint
```

## 当前实现状态

### 已完成

- [x] **top CV 预测器**（色块质心定位 → 几何法精确定心 → 色块配对优化 → 多半径弧长比法）
- [x] **side CV 预测器**（红绿颜色带投影 → 多扫描线测宽 → 圆弦几何换算，队友实现）
- [x] **CV+CNN 预测器**（CV 裁剪阀门区域 → CNN 推理，仅 side）
- [x] CNN 模型定义（MobileNetV3-Small，回归头 576→128→1）
- [x] 数据增强模块（8 种增强策略，每张原图增强 95 次）
- [x] 训练脚本（含数据增强、train/val 划分、ReduceLROnPlateau 学习率调度、断点续训、--augment 0）
- [x] 融合模块（CV + CNN 加权，CV:0.3, CNN:0.7）
- [x] 批量预测脚本（文件夹 → CSV）
- [x] **顶部视角模型已训练**（`models/mobilenetv3_top.pth`）
- [x] **侧面视角模型已训练**（`models/mobilenetv3_side.pth`）
- [x] 评估脚本 + 精度对比
- [x] 6.2 测试集 ground truth CSV 生成

### 精度

**Top 视角**（6.2 测试集，20 张）：

| 方法 | MAE |
|------|-----|
| CNN | 0.910° |
| 融合 | 1.340° |

**Side 视角**（6.2 测试集，20 张）：

| 方法 | MAE |
|------|-----|
| CNN | ~2.2° |
| CV+CNN | 待评估 |

### 待完成

- [ ] 后端服务器（视频抽帧 + 推理 API）
- [ ] 前端 App
- [ ] CV+CNN 效果优化（裁剪区域与训练数据分布差异导致效果未明显提升）

### 运行命令

```bash
# 训练顶部视角模型
python src/common/train.py --data "origin data/top" --model models/mobilenetv3_top.pth --epochs 100

# 训练侧面视角模型
python src/common/train.py --data "origin data/side" --model models/mobilenetv3_side.pth --epochs 100

# 断点续训
python src/common/train.py --data "origin data/side" --model models/mobilenetv3_side.pth --epochs 100 --resume

# 跳过增强直接训练（用于已增强的数据）
python src/common/train.py --data "finetune_data/top" --model models/mobilenetv3_top.pth --augment 0 --resume --lr 0.0001

# 批量预测（CV + CNN 融合）
python src/common/predict.py --input "origin data/top" --output output/result.csv --model models/mobilenetv3_top.pth

# 仅 CV 预测（无需模型）
python src/common/predict.py --input "origin data/top" --output output/result_cv.csv --no-cnn

# 评估精度
python src/common/evaluate.py --data "origin data/top" --model models/mobilenetv3_top.pth

# 快速测试（top 视角，CVnew + CNN + 融合 + 调试图）
python run_test_top.py

# 快速测试（side 视角，CVnew + CNN + CV+CNN + 融合）
python run_test_side.py

# 运行测试
pytest tests/ -v
```

## 环境依赖

- Python 3.x
- PyTorch ≥ 2.1（CUDA 推荐，RTX 4060 Laptop 实测可用）
- OpenCV ≥ 4.7
- torchvision
- scikit-learn
- matplotlib
- pandas
- numpy

## 参考资料

- `安特威阀门项目总结0630_edited.pptx`：老师的完整项目方案（OpenCV 方法 + AI 方法）
- `蒙板/`：拍摄时使用的蒙版模板
- `阀门测试集6.2/`：测试集数据 + 标准答案 CSV
- `新增阀门数据集6.3/`：新增训练数据（角度 6.7-46.6°，每角度 11 张混合原图+增强）
