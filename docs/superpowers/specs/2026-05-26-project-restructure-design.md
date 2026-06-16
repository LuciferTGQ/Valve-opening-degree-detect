# Project Restructure Design

## Goal

Reorganize the project to support dual-view (top/side) development. Current code is all top-view oriented with no naming distinction, scattered across root and `src/`. After restructure, each view has its own directory, shared logic lives in `common/`, and adding side view requires only filling in `src/side/` without touching existing top code.

## Target Structure

```
根目录:
├── run_test_top.py              # top 快速测试入口
├── run_test_side.py             # side 快速测试入口（待开发）
├── test_input_top/              # top 测试图片
├── test_input_side/             # side 测试图片
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── cnn_predictor.py     # CNN 模型定义 + 推理
│   │   ├── ensemble.py          # CV + CNN 加权融合
│   │   ├── data_augment.py      # 数据增强
│   │   ├── train.py             # 训练循环（通用）
│   │   └── evaluate.py          # 评估指标 + 可视化
│   ├── top/
│   │   ├── __init__.py
│   │   ├── cvnew_predictor.py   # top 视角 CV 预测器
│   │   ├── visualize_errors.py  # top 视角误差可视化
│   │   └── prepare_test.py      # top 测试数据准备
│   └── side/
│       ├── __init__.py
│       ├── cvnew_predictor.py   # side 视角 CV 预测器（待开发）
│       └── visualize_errors.py  # side 视角误差可视化（待开发）
├── models/
│   ├── mobilenetv3_top.pth
│   └── mobilenetv3_side.pth     # 未来
└── tests/
    ├── test_predict_cvnew_top.py
    ├── test_predict_cnn.py
    └── test_predict_ensemble.py
```

## File Migration Map

| 原位置 | 新位置 | 操作 |
|--------|--------|------|
| `src/cvnew_predictor.py` | `src/top/cvnew_predictor.py` | 移动 |
| `src/cnn_predictor.py` | `src/common/cnn_predictor.py` | 移动 |
| `src/ensemble.py` | `src/common/ensemble.py` | 移动 |
| `src/data_augment.py` | `src/common/data_augment.py` | 移动 |
| `src/train.py` | `src/common/train.py` | 移动 |
| `src/predict.py` | 删除 | 入口逻辑已由 run_test/evaluate 覆盖 |
| `src/compare_methods.py` | 删除 | evaluate.py 已覆盖 |
| `evaluate.py`（根目录） | 合并进 `src/common/evaluate.py` | 合并去重 |
| `visualize_errors.py`（根目录） | `src/top/visualize_errors.py` | 移动 |
| `prepare_test.py`（根目录） | `src/top/prepare_test.py` | 合并 |
| `prepare_test_with_gt.py`（根目录） | 合并进 `src/top/prepare_test.py` | 合并 |
| `run_test.py`（根目录） | `run_test_top.py` | 重命名 + 更新 import |
| `tests/test_predict_cvnew.py` | `tests/test_predict_cvnew_top.py` | 重命名 + 更新 import |
| `tests/test_predict_cnn.py` | 不变（CNN 通用） | 仅更新 import |
| `tests/test_predict_ensemble.py` | 不变 | 仅更新 import |

## Import Path Changes

### 根目录入口脚本

旧：
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from cvnew_predictor import predict_cvnew
```

新：
```python
sys.path.insert(0, os.path.dirname(__file__))
from src.top.cvnew_predictor import predict_cvnew
from src.common.cnn_predictor import predict_cnn
```

### src/ 内部文件

旧（同目录直接 import）：
```python
from cvnew_predictor import predict_cvnew
```

新（跨子包 import）：
```python
from src.top.cvnew_predictor import predict_cvnew
```

所有 `sys.path.insert` hack 统一替换为基于项目根目录的绝对 import。

## Side View 开发流程

1. 在 `src/side/cvnew_predictor.py` 实现 CV 算法（函数签名与 top 一致：`predict_cvnew(image_path) -> float`）
2. 复制 `run_test_top.py` → `run_test_side.py`，改 import 路径和目录常量
3. 复制 `src/top/prepare_test.py` → `src/side/prepare_test.py`，改数据路径
4. 训练模型：`python src/common/train.py --data "origin data/side" --model models/mobilenetv3_side.pth`

## Merge Notes

### evaluate.py 合并

根目录 `evaluate.py`（有 matplotlib 4 图可视化 + 分段指标）与 `src/compare_methods.py`（简单 CLI + `--data`/`--model` 参数）合并为 `src/common/evaluate.py`：
- 保留根目录版本的全部可视化和指标功能
- 加入 compare_methods 的 `--data`/`--model` argparse 参数
- 删除 compare_methods.py

### visualize_errors.py 接口约束

`visualize_errors.py` 的 `predict_with_debug()` 直接调用 cvnew_predictor 的内部函数（`_filter_green`、`_filter_red`、`_refine_center` 等）。每个视角的 `cvnew_predictor.py` 必须导出这些函数，供 visualize_errors 使用。

## Constraints

- 不改变任何算法逻辑，纯文件重组
- 所有现有测试必须通过
- `run_test_top.py` 功能与原 `run_test.py` 完全一致
