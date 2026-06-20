# 手机 App 视频上传与分析接口说明

## 1. 整体通信流程

手机 App 不直接访问运行 Python 的电脑，而是访问 ECS 的公网地址：

```text
手机 App
  -> 47.110.89.148:9000
  -> ECS 上的 frps
  -> 电脑上的 frpc
  -> 127.0.0.1:8000
  -> Python HTTP 服务
```

端口作用：

| 端口 | 所在位置 | 作用 |
| --- | --- | --- |
| `7000` | ECS | frpc 与 frps 建立连接的控制端口 |
| `9000` | ECS | 手机 App 上传视频的公网业务端口 |
| `8000` | 本地电脑 | Python HTTP 服务监听端口 |

端口转发关系：

```text
ECS:9000 -> 本地电脑:8000
```

手机不应访问 `7000` 或 `8000`，只访问 ECS 公网的 `9000` 端口。

## 2. 启动顺序

### 2.1 启动 Python 后端

在项目根目录执行：

```powershell
python src\video_http_server.py --host 127.0.0.1 --port 8000
```

启动成功后会显示：

```text
Valve video HTTP server listening on http://127.0.0.1:8000
POST endpoint: /predict-video
```

### 2.2 启动 frpc

确认 `tools/frp/frpc.toml` 中的端口映射为：

```toml
serverAddr = "47.110.89.148"
serverPort = 7000

[[proxies]]
name = "valve-video-upload"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8000
remotePort = 9000
```

然后启动 frpc：

```powershell
powershell -ExecutionPolicy Bypass -File tools\frp\start_frpc.ps1
```

### 2.3 检查公网连接

```powershell
curl.exe http://47.110.89.148:9000/health
```

成功时返回类似：

```json
{
  "ok": true,
  "service": "valve-video-predictor",
  "supported_algorithms": [
    "side_cnn",
    "side_cv",
    "side_fusion",
    "top_cnn",
    "top_cv"
  ]
}
```

## 3. 上传接口

请求地址：

```text
POST http://47.110.89.148:9000/predict-video
```

请求格式：

```text
multipart/form-data
```

必须包含以下表单字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `video` | 文件 | 需要分析的视频 |
| `algorithm` | 文本 | 需要使用的识别算法 |

可用算法：

| algorithm | 说明 |
| --- | --- |
| `side_cv` | 侧面 CV 算法 |
| `side_cnn` | 侧面 CNN 算法 |
| `side_fusion` | 侧面 CV/CNN 状态机融合算法 |
| `top_cv` | 顶部 CV 算法 |
| `top_cnn` | 顶部 CNN 算法 |

## 4. 通用可选参数

| 字段 | 默认值 | 说明 |
| --- | ---: | --- |
| `frame_stride` | `1` | 每隔多少帧分析一次；`1` 表示每帧分析 |
| `max_frames` | 不限制 | 最多分析多少个抽取帧 |
| `model_path` | 默认模型 | 可选的服务端 CNN 模型路径 |
| `strict` | `false` | 单帧失败时是否立即中止整个请求 |

例如：

```text
frame_stride=5
max_frames=100
```

表示每 5 帧分析一次，最多输出 100 条结果。

## 5. Side Fusion 参数

当 `algorithm=side_fusion` 时，还可以传入：

| 字段 | 默认值 | 说明 |
| --- | ---: | --- |
| `fusion_low` | `10` | 小角度中心阈值 |
| `fusion_high` | `70` | 大角度中心阈值 |
| `fusion_transition` | `5` | Smoothstep 过渡带总宽度 |
| `fusion_cv_probe_interval` | `15` | CNN 区间每隔多少帧复查一次 CV；`0` 表示关闭周期复查 |
| `fusion_probe_margin` | `5` | CNN 接近阈值多少度时强制复查 CV |

融合状态：

```text
较小角度 -> CV_ONLY
中间角度 -> CNN_ONLY
较大角度 -> CV_ONLY
阈值过渡带 -> CV 与 CNN 平滑混合
```

## 6. Side Fusion 上传示例

使用 `curl.exe` 模拟手机 App 上传：

```powershell
curl.exe -X POST "http://47.110.89.148:9000/predict-video" `
  -F "video=@test_video\3.mp4;type=video/mp4" `
  -F "algorithm=side_fusion" `
  -F "fusion_low=10" `
  -F "fusion_high=70" `
  -F "fusion_transition=5" `
  -F "fusion_cv_probe_interval=15" `
  -F "fusion_probe_margin=5"
```

后端将执行：

```text
保存视频
-> 判断每帧所处角度区间
-> 按需运行 side_cv 或 side_cnn
-> 生成融合结果
-> 生成 CSV 和 PNG
-> 返回 JSON
```

## 7. Top CV 上传示例

```powershell
curl.exe -X POST "http://47.110.89.148:9000/predict-video" `
  -F "video=@test_video\4.mp4;type=video/mp4" `
  -F "algorithm=top_cv"
```

如果只想快速测试前 20 帧：

```powershell
curl.exe -X POST "http://47.110.89.148:9000/predict-video" `
  -F "video=@test_video\4.mp4;type=video/mp4" `
  -F "algorithm=top_cv" `
  -F "max_frames=20"
```

## 8. Side Fusion 返回示例

后端处理完整段视频后返回 JSON：

```json
{
  "ok": true,
  "request_id": "85b1f3b80da04fb39eb26ce842659e31",
  "algorithm": "side_fusion",
  "video_path": "D:/.../uploads/videos/upload.mp4",
  "csv_path": "D:/.../output/video_http/result_side_fusion.csv",
  "plot_path": "D:/.../output/video_http/result_side_fusion.png",
  "frame_count": 174,
  "failed_count": 0,
  "stiction_level": 2,
  "elapsed_ms": 29341.9,
  "results": [
    {
      "frame_index": 0,
      "timestamp_ms": 0.0,
      "cv_angle": 75.9,
      "cnn_angle": null,
      "fusion_angle": 75.9,
      "selected_algorithm": "side_cv",
      "cv_error": null,
      "cnn_error": null
    }
  ]
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `fusion_angle` | 手机 App 应使用的最终阀门开度 |
| `selected_algorithm` | 当前帧实际采用的算法或混合状态 |
| `cv_angle` | 当前帧的 CV 结果；跳过 CV 时为 `null` |
| `cnn_angle` | 当前帧的 CNN 结果；跳过 CNN 时为 `null` |
| `cv_error` | CV 错误信息；成功时为 `null` |
| `cnn_error` | CNN 错误信息；成功时为 `null` |
| `stiction_level` | 整段视频的卡涩程度，取值为 `1`、`2` 或 `3`，数字越大表示越卡涩 |

`selected_algorithm` 可能为：

```text
side_cv
side_cnn
smooth_blend_low
smooth_blend_high
side_cnn_fallback
```

## 9. 普通算法返回示例

使用 `side_cv`、`side_cnn`、`top_cv` 或 `top_cnn` 时，每帧结果结构为：

```json
{
  "frame_index": 0,
  "timestamp_ms": 0.0,
  "angle": 38.9,
  "algorithm": "top_cv",
  "error": null
}
```

## 10. App 实际需要使用的数据

对于 `side_fusion`，App 主要使用：

```text
results[].timestamp_ms
results[].fusion_angle
stiction_level
```

对于其他算法，App 主要使用：

```text
results[].timestamp_ms
results[].angle
stiction_level
```

`stiction_level` 位于返回 JSON 顶层，是对整段视频的估计，不是单帧字段：

```text
1 = 运行较顺畅
2 = 存在一定卡涩
3 = 卡涩较严重
```

当前卡涩分级使用停滞、突跳和反向抖动的启发式评分，后续需要使用真实三级标注视频校准。

`video_path`、`csv_path` 和 `plot_path` 是运行 Python 服务的电脑上的本地路径，手机不能直接打开。如果 App 需要下载 CSV 或 PNG，还需要后端提供文件下载接口。

## 11. 等待时间与超时

当前接口采用同步处理：

```text
上传完成 -> 后端分析完整段视频 -> 返回 JSON
```

分析期间 HTTP 请求会保持等待。手机 App 的 HTTP 超时时间建议设置为：

```text
至少 300 秒
```

视频较长或使用 CNN 时，应适当增加超时时间。

## 12. 常见问题

### 公网健康检查失败

依次检查：

```text
Python 服务是否监听 127.0.0.1:8000
frpc 是否成功连接 ECS:7000
frpc 是否配置 remotePort=9000
ECS 安全组是否放行 7000 和 9000
```

### 返回 404

真实分析服务的地址必须是：

```text
/predict-video
```

`/upload-video` 是假处理 Demo 的接口，不用于真实算法分析。

### 返回 unsupported algorithm

确认算法名拼写正确，例如：

```text
side_fusion
```

旧名称 `cnn_cv_fusion` 已不再支持。
