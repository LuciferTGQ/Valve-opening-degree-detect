# Side 视角 CV+CNN 训练优化设计

日期: 2026-05-28

## 背景

Side 视角 CNN 模型效果一般（MAE ~3.5°），而 Top 已经不错（MAE ~0.9°）。
现有 cvcnn_predictor 推理时先 CV 裁剪阀门区域再 CNN 预测，但训练时 CNN
用的是完整图片（含背景干扰）。本次优化让训练数据也先裁剪，使训练和推理输入分布一致。

## 步骤

### 1. 数据准备

- 把 `新增阀门数据集6.3/side/` (220) + `新增阀门数据集6.10/side/` (220) 复制进 `origin data/side/`
- 总计 598 张原图
- 删掉 `origin data/data_augmented/side/`，重新全量增强 95 倍 → ~57k 张

### 2. CV 裁剪训练数据（新脚本）

写 `src/side/crop_augmented.py`：
- 遍历 `origin data/data_augmented/side/` 的每张增强图
- 用 `_crop_valve_region()` 裁剪阀门区域
- 裁剪成功的保存到 `data_cropped/side/`（同文件名，保留角度解析）
- 裁剪失败的跳过（不保留原图，保持训练数据一致性）
- 预计产出 ~55k+ 张

### 3. 训练新模型

```bash
python src/common/train.py --data "data_cropped/side" --augment 0 \
  --model models/mobilenetv3_side_cv.pth --epochs 100
```

- `--augment 0`：数据已增强，直接读取裁剪图
- 新模型保存为 `mobilenetv3_side_cv.pth`
- 旧模型 `mobilenetv3_side.pth` 不动

### 4. 统一测试集

合并三个测试集：
- `阀门测试集6.2/side/` (20) + CSV
- `阀门测试集6.9/side/` (20) + CSV
- `阀门测试集6.16/side/` (20) + CSV
→ `test_input_side_merged/` (60 张) + `ground_truth_side_merged.csv`

### 5. 评估对比

| 模型 | 推理方式 | 代码 |
|------|----------|------|
| 旧模型 | 直接 CNN | `predict_cnn(image, mobilenetv3_side.pth)` |
| 新模型 | CV裁剪 + CNN | `predict_cvcnn(image, mobilenetv3_side_cv.pth)` |

对比 60 张测试集的 MAE/RMSE。新模型更好时考虑对 top 同样优化。

## GPU 利用

train.py 已支持 CUDA + num_workers=4 + pin_memory=True。增强是 OpenCV CPU 操作，
无法 GPU 加速（瓶颈是磁盘 I/O 不是计算）。

## 不做的事情

- 不覆盖旧模型
- 不修改 top 视角（等 side 结果再决定）
- 不修改 train.py 核心逻辑（只新增裁剪脚本）
