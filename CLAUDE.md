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

1. **CV 方法（cvnew_predictor）**：HoughCircles 定位圆 → 色块质心 + 几何法精确定位圆心 → 多半径弧长比法计算角度
2. **CNN 方法（cnn_predictor）**：MobileNetV3-Small 回归模型，输入单帧图像 → 输出角度
3. **融合（ensemble）**：CV:0.3 + CNN:0.7 加权融合

### 数据增强策略

- 水平翻转（50%）、垂直翻转（30%）
- 亮度调节（0.7-1.3x）
- 对比度调节（0.75-1.25x）
- 高斯模糊（kernel 3/5/7，50%）
- 旋转（-15° ~ +15°）

## 数据结构

### 目录布局

```
origin data/
├── top/                    # 顶部视角图像（44张，jpg）
│   └── NNNN_angle.jpg      # 如 0001_3.4.jpg
├── side/                   # 正侧面视角图像（128张，jpg+png混合）
│   ├── NNNN_angle.jpg      # 38张，如 0001_8.2.jpg
│   └── p_N_angle.png       # 90张，如 p_1_1.0.png
├── data_augmented/
│   └── top/                # 增强后的顶部图像（484张 = 44原图 × 11变体）
├── video/                  # 原始视频文件
│   ├── *.mp4               # 顶部和侧面拍摄视频
│   ├── side1/              # 侧面1抽帧结果（72张，1-80°）
│   │   └── p_N_angle.png
│   └── side2/              # 侧面2抽帧结果（90张，1-80°）
│       └── p_N_angle.png
└── 蒙板/                   # 拍摄蒙版模板
    ├── 红绿1.png ~ 红绿3.png
    ├── 红黄1.png ~ 红黄3.png
    └── 圆形阀门/            # 圆形阀门蒙版（上/侧视角）

unlabbled_top/              # 未标注的顶部视角照片（57张）
unlabbled_side/             # 未标注的侧面视角照片（114张）
```

### 文件命名规范

- `top/` 和 `side/`（jpg部分）：`NNNN_angle.jpg`，序号 4 位零填充，角度精确到 1 位小数
- `side/`（png部分）和 `video/*/`：`p_N_angle.png`，序号无零填充
- 角度范围：0.0-80.0 度（全绿=80°，全红=0°）

### 关键数据量

| 目录 | 视角 | 数量 | 格式 |
|------|------|------|------|
| top/ | 顶部 | 44张 | jpg |
| side/ | 正侧面 | 128张 | jpg+png |
| data_augmented/top/ | 顶部增强 | 484张 | jpg |
| video/side1/ | 侧面1帧 | 72张 | png |
| video/side2/ | 侧面2帧 | 90张 | png |
| unlabbled_top/ | 未标注顶部 | 57张 | jpg |
| unlabbled_side/ | 未标注侧面 | 114张 | jpg |

## 阀门类型

本项目**只关注 top 和 side 两种视角**，只关注**红绿阀门**：

1. **红绿阀门**：顶部红绿指示器，圆柱体主体

## 核心约束

- **输出精度**：0.1 度
- **角度范围**：0-80 度
- **全绿 = 80°，全红 = 0°**
- 只考虑红绿阀门，且只考虑正上方和正侧方，训练两个独立模型
- 数据增强是必需的（原始数据量不足以泛化真实场景）
- CV 方法在极低角度（<5°）时绿色饱和度过低，难以检测（固有盲区）

## 代码结构

```
src/
├── cvnew_predictor.py  # CV 预测器（HoughCircles + 色块质心 + 几何法 + 弧长比法）
├── cnn_predictor.py    # CNN 模型定义（MobileNetV3-Small）+ 单帧推理
├── ensemble.py         # CV + CNN 加权融合（默认 CV:0.3, CNN:0.7）
├── data_augment.py     # 数据增强（翻转、亮度、对比度、模糊、旋转）
├── train.py            # 训练脚本（数据增强 → 划分 → 训练 → 保存最佳模型）
├── predict.py          # 批量预测入口（文件夹 → CSV）
└── compare_methods.py  # 方法精度对比工具
evaluate.py             # 评估脚本（MAE/RMSE/分段指标 + matplotlib 可视化）
prepare_test.py         # 工具：随机采样图片到 test_input/
prepare_test_with_gt.py # 工具：采样图片 + 保存 ground truth CSV
run_test.py             # 运行 CV/CNN/Ensemble 三种方法并对比
tests/
├── test_predict_cvnew.py     # CV 预测测试
├── test_predict_cnn.py       # CNN 预测测试
└── test_predict_ensemble.py  # 融合预测测试
models/
└── mobilenetv3_top.pth       # 已训练的顶部视角模型（~4MB）
```

## 当前实现状态

### 已完成

- [x] CV 预测器（HoughCircles 定位 → 色块质心 + 几何法精确定心 → 多半径弧长比法）
- [x] CNN 模型定义（MobileNetV3-Small，回归头 576→128→1）
- [x] 数据增强模块（6 种增强策略）
- [x] 训练脚本（含数据增强、train/val 划分、ReduceLROnPlateau 学习率调度）
- [x] 融合模块（CV + CNN 加权，CV:0.3, CNN:0.7）
- [x] 批量预测脚本（文件夹 → CSV）
- [x] **顶部视角模型已训练**（`models/mobilenetv3_top.pth`）
- [x] 评估脚本 + 精度对比

### 顶部视角精度（44张，CVnew 方法）

| 指标 | 值 |
|------|-----|
| MAE | 1.15° |
| RMSE | 1.39° |
| 最大误差 | 3.40° |

- 0-20°：MAE 1.07（15张）
- 20-40°：MAE 1.23（18张）
- 40-60°：MAE 1.48（4张）
- 60-80°：MAE 1.10（6张）

### 待完成

- [ ] 侧面视角模型训练（`origin data/side/` 数据已就绪，128张）
- [ ] 后端服务器（视频抽帧 + 推理 API）
- [ ] 前端 App
- [ ] CNN 顶部模型精度评估（需对比 ground truth）

### 运行命令

```bash
# 训练顶部视角模型
python src/train.py --data "origin data/top" --model models/mobilenetv3_top.pth --epochs 100

# 训练侧面视角模型
python src/train.py --data "origin data/side" --model models/mobilenetv3_side.pth --epochs 100

# 批量预测（CV + CNN 融合）
python src/predict.py --input "origin data/top" --output output/result.csv --model models/mobilenetv3_top.pth

# 仅 CV 预测（无需模型）
python src/predict.py --input "origin data/top" --output output/result_cv.csv --no-cnn

# 评估精度
python evaluate.py

# 对比方法精度
python src/compare_methods.py --data "origin data/top" --model models/mobilenetv3_top.pth

# 运行测试
pytest tests/ -v
```

## 环境依赖

- Python 3.x
- PyTorch ≥ 2.1
- OpenCV ≥ 4.7
- torchvision
- scikit-learn
- matplotlib
- pandas
- numpy

## 参考资料

- `安特威阀门项目总结0630_edited.pptx`：老师的完整项目方案（OpenCV 方法 + AI 方法）
- `蒙板/`：拍摄时使用的蒙版模板
