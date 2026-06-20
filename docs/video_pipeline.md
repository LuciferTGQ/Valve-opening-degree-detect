# 视频上传与逐帧分析

## 1. 本地假处理联调

启动只保存视频、等待并返回假结果的服务：

```powershell
python src/video_upload_demo.py --host 127.0.0.1 --port 8000 --fake-delay 3 --fake-frame-count 10
```

使用 Python 客户端模拟 Unity 上传：

```powershell
python src/video_upload_client.py --url http://127.0.0.1:8000/upload-video --video path\to\test.mp4
```

## 2. 真实逐帧分析

启动分析服务：

```powershell
python src/video_http_server.py --host 127.0.0.1 --port 8000
```

上传视频并指定算法：

```powershell
python src/video_upload_client.py `
  --url http://127.0.0.1:8000/predict-video `
  --video path\to\test.mp4 `
  --algorithm side_cv `
  --frame-stride 1 `
  --output-json output\video_response.json
```

支持的算法：`side_cv`、`side_cnn`、`top_cv`、`top_cnn`。

## 3. frpc

1. 将 `tools/frp/frpc.example.toml` 复制为 `tools/frp/frpc.toml`。
2. 把 `auth.token` 修改为 ECS 上 frps 使用的新 token。
3. 启动 Python HTTP 服务。
4. 运行：

```powershell
powershell -ExecutionPolicy Bypass -File tools\frp\start_frpc.ps1
```

当前端口映射：

```text
Unity -> 47.110.89.148:9000 -> frps:7000 -> frpc -> 127.0.0.1:8000
```

公网测试地址：

```text
GET  http://47.110.89.148:9000/health
POST http://47.110.89.148:9000/upload-video
POST http://47.110.89.148:9000/predict-video
```

## 4. 自动测试

```powershell
python -m unittest tests.test_video_upload_pipeline -v
```

测试会自动创建短视频，并验证：视频上传保存、假结果返回、`top_cv` 逐帧分析和 CSV 输出。
