# 视频测试控制台使用说明

## 1. 进入项目目录

打开 PowerShell 或 CMD，进入项目根目录：

```powershell
cd "D:\01_EGGER\program\python\Valve-opening-degree-detect"
```

所有命令都应在项目根目录中执行。

## 2. 基本测试命令

使用侧面 OpenCV 算法分析 `test_video/3.mp4`：

```powershell
python src\video_predictor.py --video test_video\3.mp4 --algorithm side_cv --output output\3_side_cv.csv
```

处理过程中，控制台会逐帧输出：

```text
[1/174] 帧 0, 时间 0.000s, 开度 75.9° (0.261s)
[2/174] 帧 1, 时间 0.033s, 开度 75.7° (0.257s)
```

处理结束后会生成：

```text
output/3_side_cv.csv
output/3_side_cv.png
```

CSV 保存每一帧的预测值，PNG 是阀门开度随视频时间变化的曲线。

## 3. 算法选择

`--algorithm` 支持四种算法：

| 参数 | 说明 | 默认模型 |
| --- | --- | --- |
| `side_cv` | 侧面 CVnew 几何算法 | 不需要模型 |
| `side_cnn` | 侧面 CV 裁剪加 CNN | `models/mobilenetv3_side_cv.pth` |
| `top_cv` | 顶部 CVnew 算法 | 不需要模型 |
| `top_cnn` | 顶部 CNN 算法 | `models/mobilenetv3_top.pth` |

侧面 CNN 示例：

```powershell
python src\video_predictor.py --video test_video\3.mp4 --algorithm side_cnn --output output\3_side_cnn.csv
```

顶部 CV 示例：

```powershell
python src\video_predictor.py --video test_video\4.mp4 --algorithm top_cv --output output\4_top_cv.csv
```

顶部 CNN 示例：

```powershell
python src\video_predictor.py --video test_video\4.mp4 --algorithm top_cnn --output output\4_top_cnn.csv
```

## 4. 指定曲线图路径

默认情况下，曲线图与 CSV 使用相同文件名。也可以通过 `--plot` 单独指定：

```powershell
python src\video_predictor.py --video test_video\3.mp4 --algorithm side_cv --output output\result.csv --plot output\result_curve.png
```

## 5. 抽帧测试

每隔 5 帧分析一次，可以明显缩短处理时间：

```powershell
python src\video_predictor.py --video test_video\3.mp4 --algorithm side_cv --frame-stride 5 --output output\3_stride5.csv
```

参数含义：

```text
--frame-stride 1    每一帧都分析
--frame-stride 2    每隔两帧分析一次
--frame-stride 5    每隔五帧分析一次
```

## 6. 只测试前几帧

只分析前 20 个被抽取的帧：

```powershell
python src\video_predictor.py --video test_video\3.mp4 --algorithm side_cv --max-frames 20 --output output\3_preview.csv
```

快速检查算法是否能运行时，建议先使用 `--max-frames 5`。

## 7. 同时使用抽帧和帧数限制

下面的命令每隔 5 帧分析一次，最多分析 20 帧：

```powershell
python src\video_predictor.py --video test_video\3.mp4 --algorithm side_cv --frame-stride 5 --max-frames 20 --output output\3_quick_test.csv
```

## 8. 指定 CNN 模型

如需覆盖默认模型路径，可以使用 `--model`：

```powershell
python src\video_predictor.py --video test_video\3.mp4 --algorithm side_cnn --model models\mobilenetv3_side_cv.pth --output output\3_custom_model.csv
```

## 9. 控制错误处理与控制台输出

默认情况下，单帧识别失败会写入 CSV 的 `error` 字段，并继续处理后续帧。

使用 `--strict` 后，任意一帧失败都会立即终止：

```powershell
python src\video_predictor.py --video test_video\3.mp4 --algorithm side_cv --strict --output output\3_strict.csv
```

不显示逐帧分析过程：

```powershell
python src\video_predictor.py --video test_video\3.mp4 --algorithm side_cv --quiet --output output\3_quiet.csv
```

## 10. CSV 字段

输出 CSV 包含以下字段：

| 字段 | 说明 |
| --- | --- |
| `frame_index` | 视频中的原始帧编号，从 0 开始 |
| `timestamp_ms` | 当前帧对应的视频时间，单位毫秒 |
| `angle` | 阀门开度预测值，范围 0 到 80 度 |
| `algorithm` | 当前使用的算法 |
| `error` | 单帧分析错误；成功时为空 |

## 11. 批量测试 test_video 中的视频

使用侧面 CV 算法批量分析所有 MP4：

```powershell
Get-ChildItem test_video\*.mp4 | ForEach-Object {
    python src\video_predictor.py `
        --video $_.FullName `
        --algorithm side_cv `
        --output "output\$($_.BaseName)_side_cv.csv"
}
```

## 12. 查看所有参数

```powershell
python src\video_predictor.py --help
```

## 13. CV 与 CNN 同图对比

使用 `side_compare` 可以在同一批侧面视频帧上同时运行 CV 和 CNN：

```powershell
python src\video_predictor.py --video test_video\3.mp4 --algorithm side_compare --output output\video3_side_compare.csv
```

生成的 PNG 会在同一张图中用蓝色表示 CV、红色表示 CNN。CSV 包含：

```text
frame_index,timestamp_ms,cv_angle,cnn_angle,abs_difference,cv_error,cnn_error
```

顶部视角使用：

```powershell
python src\video_predictor.py --video test_video\4.mp4 --algorithm top_compare --output output\video4_top_compare.csv
```

## 14. Side CNN-CV 融合模式

融合模式名称为 `side_fusion`。默认使用侧面 CV 的结果判断角度区间：

```text
CV <= 10°       使用 side_cv
10° < CV < 70°  使用 side_cnn
CV >= 70°       使用 side_cv
```

为避免硬切换跳变，默认在 10°和70°两个中心阈值附近使用 Smoothstep 平滑加权，过渡带总宽度为 `5°`。

融合模式采用状态机按需计算：

- `CV_ONLY`：较小和较大角度只运行 CV，跳过 CNN。
- `CNN_ONLY`：CV 明确确认进入中间区后，只运行 CNN。
- CNN 接近阈值时强制复查 CV，默认每 15 个 CNN 帧也周期复查一次，避免因 CNN 偏差错过跨区。
- 过渡带同时运行 CV 和 CNN，并进行 Smoothstep 混合。

运行命令：

```powershell
python src\video_predictor.py --video test_video\3.mp4 --algorithm side_fusion --output output\video3_side_fusion.csv
```

输出 CSV 同时保存 `cv_angle`、`cnn_angle`、`fusion_angle` 和 `selected_algorithm`。PNG 中蓝线为 CV、红线为 CNN、绿线为最终融合结果。

可以调整小角度和大角度的分界值：

```powershell
python src\video_predictor.py `
    --video test_video\3.mp4 `
    --algorithm side_fusion `
    --fusion-low 15 `
    --fusion-high 65 `
    --fusion-transition 8 `
    --fusion-cv-probe-interval 15 `
    --fusion-probe-margin 5 `
    --output output\video3_fusion_15_65.csv
```

将 `--fusion-transition` 设置为 `0` 可以恢复硬切换。

将 `--fusion-cv-probe-interval` 设置为 `0` 会关闭 CNN 区间的周期 CV 复查，但仍会在 CNN 接近阈值时复查。这样计算量更低，但在 CNN 偏差较大时存在延迟切换风险。

目前融合模式只用于 side，top 暂不参与融合。

## 15. 卡涩程度测试

使用 `--stiction-test` 可以在视频分析完成后估计整段阀门运动的卡涩程度：

```powershell
python src\video_predictor.py `
    --video test_video\3.mp4 `
    --algorithm side_fusion `
    --stiction-test `
    --output output\video3_side_fusion.csv
```

卡涩程度分为三级：

| 等级 | 含义 |
| ---: | --- |
| `1` | 运行较顺畅 |
| `2` | 存在一定停滞、突跳或反向抖动 |
| `3` | 存在明显的停滞-突跳现象，卡涩较严重 |

控制台会输出：

```text
卡涩程度: 2 级
卡涩评分: 0.1810
有效行程: 78.400°
运动时长: 8.900s
停滞占比: 0.2022
突跳占比: 0.2326
反向抖动占比: 0.0000
```

估计器会先使用 5 帧中值滤波去除单帧尖峰，并排除行程起点和终点的正常停留，然后综合停滞占比、突跳运动占比和反向抖动占比进行分级。

当前分级阈值是启发式参数，正式使用前应使用带有 1/2/3 级真实卡涩标注的视频进行校准。

出现模型不存在、视频无法打开或模块导入失败时，应先确认命令运行目录是项目根目录，并检查视频和模型路径是否存在。
