# 阀门开度识别系统 (Valve Opening Degree Detection)

## 项目概述

安特威阀门有限公司 × 南京航空航天大学 IoT 课设项目。通过计算机视觉技术，从阀门视频中提取每一帧，识别开度指示器角度（0-80°，精度 0.1°），进而判断阀门卡涩程度。

卡涩程度判定函数由安特威提供（黑盒），我们只需**输入图片 → 输出开度角度**，函数自行判断卡涩等级。

## 系统架构

```
移动端 App → 拍摄视频 → 上传服务器 → 抽帧 → CNN推理 → 角度序列 → 卡涩判定 → 返回结果
```

- **前端**：移动端 App，支持拍摄视频、上传、查看结果
- **后端**：Python 服务器，负责视频处理和 AI 推理
- **算法核心**：CNN 回归模型，输入单帧图像，输出开度角度（0.0-80.0）

## 技术方案

### AI 方法（本项目采用）

与老师的 OpenCV 方法不同，本项目采用 CNN 回归方法：

1. **数据采集**：从原始视频中均匀抽帧，人工标注角度
2. **数据增强**：模拟真实场景（亮度、对比度、饱和度调整，模糊，旋转，微调等）
3. **模型训练**：每种阀门的每种视角（top/side）训练一个独立模型
4. **模型推理**：输入单帧图像 → 输出角度预测值

### 参考模型（PPT 中测试过的）

| 模型 | 参数量 | 速度（90张） | 特点 |
|------|--------|-------------|------|
| MobileNetV3-Small | 最少 | 最快 | 轻量级 |
| MobileNetV3-Large | 中等 | ~7.5s | 速度精度平衡好（推荐） |
| ResNet32 | 较多 | 较慢 | - |
| ResNet50 | 最多 | ~38s | 精度最高但最慢 |

### 数据增强策略（参考 PPT）

- 亮度调节（模拟光照变化）
- 对比度调节
- 饱和度调节
- 远距离模糊（模拟不同拍摄距离）
- 左右旋转（模拟拍摄角度偏移）
- 顺/逆时针自旋 0-10°（模拟手抖）

## 数据结构

### 目录布局

```
origin data/
├── top/                    # 顶部视角图像（44张）
│   └── NNNN_angle.jpg      # 如 0001_3.4.jpg（序号_角度）
├── side/                   # 正侧面视角图像（128张）
│   └── NNNN_angle.jpg      # 如 0001_8.2.jpg（序号_角度）
├── other/                  # 其他视角/增强数据（180张）
│   └── p_NNN_angle.png     # 如 p_1_3.4.png
├── video/                  # 原始视频文件
│   ├── *.mp4               # 顶部和侧面拍摄视频
│   ├── side1/              # 侧面1抽帧结果（72张，1-80°）
│   │   └── p_N_angle.png
│   └── side2/              # 侧面2抽帧结果（90张，1-80°）
│       └── p_N_angle.png
├── 红绿阀门双层/            # 红绿阀门原始照片（3人各95张）
│   ├── nrb/
│   ├── wqq/
│   └── zzy/
├── 蒙板/                   # 拍摄蒙版模板
│   ├── 红绿1.png ~ 红绿3.png
│   ├── 红黄1.png ~ 红黄3.png
│   └── 圆形阀门/            # 圆形阀门蒙版
└── 安特威阀门项目总结0630_edited.pptx  # 参考PPT
```


### 文件命名规范

- `top/` 和 `side/`：`NNNN_angle.jpg`，序号 4 位零填充，角度精确到 1 位小数
- `other/` 和 `video/*/`：`p_NNN_angle.png`，序号无零填充
- 角度范围：0.0-80.0 度（全绿=80°，全红=0°）

### 关键数据量

| 目录 | 视角 | 数量 | 格式 |
|------|------|------|------|
| top/ | 顶部 | 44张 | jpg |
| side/ | 正侧面 | 128张 | jpg |
| other/ | 其他 | 180张 | png |
| video/side1/ | 侧面1帧 | 72张 | png |
| video/side2/ | 侧面2帧 | 90张 | png |

## 阀门类型

本项目**只关注 top 和 side 两种视角**，只关注**红绿阀门**：

1. **红绿阀门**：顶部红绿指示器，圆柱体主体


## 核心约束

- **输出精度**：0.1 度
- **角度范围**：0-80 度
- **全绿 = 80°，全红 = 0°**
- 只考虑红绿阀门，且只考虑正上方和正侧方，训练两个独立模型。
- 数据增强是必需的（原始数据量不足以泛化真实场景）
- 拍摄稳定性影响推理效果（App 端需引导用户稳定拍摄）

## 项目阶段

1. **数据准备**：整理原始数据 → 标注 → 数据增强
2. **模型开发**：选择架构 → 训练 → 验证
3. **后端开发**：视频抽帧 → 模型推理 → 角度序列输出
4. **前端开发**：App 拍摄 → 上传 → 结果展示
5. **集成测试**：端到端流程验证

## 当前实现状态

### 代码结构

```
src/
├── cnn_predictor.py    # CNN 模型定义 + 单帧推理
├── cv_predictor.py     # OpenCV 方法（HoughCircles + 绿色占比）
├── data_augment.py     # 数据增强（翻转、亮度、对比度、模糊、旋转）
├── ensemble.py         # CV + CNN 加权融合（默认 CV:0.3, CNN:0.7）
├── train.py            # 训练脚本（数据增强 → 划分 → 训练 → 保存最佳模型）
└── predict.py          # 主入口：批量预测文件夹 → 输出 CSV
tests/
├── test_cv_predictor.py
└── test_predict.py
models/
└── mobilenetv3_top.pth # 已训练的顶部视角模型（MobileNetV3-Small）
```

### 已完成

- [x] CV 预测器（HoughCircles + HSV 绿色占比）
- [x] CNN 模型定义（MobileNetV3-Small，回归头 576→128→1）
- [x] 数据增强模块（6 种增强策略）
- [x] 训练脚本（含数据增强、train/val 划分、学习率调度）
- [x] 融合模块（CV + CNN 加权）
- [x] 批量预测脚本（文件夹 → CSV）
- [x] **顶部视角模型已训练**（`models/mobilenetv3_top.pth`）

### 待完成

- [ ] 侧面视角模型训练（`origin data/side/` 数据已就绪）
- [ ] 后端服务器（视频抽帧 + 推理 API）
- [ ] 前端 App
- [ ] 精度评估（需要 ground truth 对比）

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

# 运行测试
pytest tests/ -v
```

## 环境依赖（参考）

- Python 3.x
- PyTorch ≥ 2.1
- OpenCV ≥ 4.7
- torchvision
- scikit-learn
- matplotlib
- scipy

## 参考资料

- `安特威阀门项目总结0630_edited.pptx`：老师的完整项目方案（OpenCV 方法 + AI 方法）
- `阀门开度识别样例数据.zip`：样例数据
- `蒙板/`：拍摄时使用的蒙版模板
