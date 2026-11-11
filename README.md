# 高拍仪扫描件矫正服务

对上传的扫描件自动做：纸张检测与透视矫正 → 去倾斜 → 90° 方向矫正（证据充分时修正倒置）→ 裁黑边与多余背景。纯 OpenCV + numpy，不含模型，Python 3.8。

| 文件 | 说明 |
|------|------|
| `app.py` | FastAPI 服务：页面、处理接口、健康检查 |
| `image_processor.py` | 核心算法，可调参数集中在文件头部；也可命令行单独运行 |
| `static/index.html` | 上传对比页面，不依赖 CDN |
| `requirements.txt` | 固定版本依赖 |
| `scripts/build_offline.sh` | 联网机器上制作离线包 |
| `scripts/install_offline.sh` | 现场服务器离线安装（随离线包分发） |

## 环境要求

- Linux x86_64 / aarch64，glibc ≥ 2.17（CentOS 7 及以上）
- **Python 3.8**（离线包内 wheel 均为 cp38，不含 Python 本身）
- 打包机与现场服务器 CPU 架构一致

## 部署流程

部署机器（联网）安装验证并打离线包 → 拷到现场服务器（不联网）离线安装。

### 1. 联网机器：安装并验证

```bash
python3.8 -m venv venv
venv/bin/python -m pip install -r requirements.txt   # 可加 -i https://pypi.tuna.tsinghua.edu.cn/simple
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000   # 必须在工程目录下启动
```

浏览器打开 `http://<IP>:8000/` 上传扫描件对比效果。只测算法：`venv/bin/python image_processor.py 输入.jpg 输出.jpg`。

### 2. 打离线包

```bash
bash scripts/build_offline.sh                     # 目标 x86_64
TARGET_ARCH=aarch64 bash scripts/build_offline.sh # 目标 ARM（鲲鹏/飞腾）
```

产物为 `dist/image-process-offline.tar.gz` 和 `.sha256`。拷走前建议先在本机离线装一遍（全程 `--no-index`，装成功即说明包完整）：

```bash
cd /tmp && tar -xzf /path/to/dist/image-process-offline.tar.gz && cd image-process-offline
INSTALL_DIR=/tmp/imgproc-verify bash install_offline.sh
```

### 3. 现场服务器：离线安装

```bash
sha256sum -c image-process-offline.tar.gz.sha256
tar -xzf image-process-offline.tar.gz && cd image-process-offline
sudo bash install_offline.sh 2>&1 | tee install.log
```

| 环境变量 | 默认值 | 说明 |
|------|--------|------|
| `INSTALL_DIR` | `/opt/image-process` | 安装目录；无 sudo 时设为 `$HOME/image-process` |
| `PYTHON_BIN` | 自动查找 `python3.8`、`python3` | 如 `/opt/rh/rh-python38/root/usr/bin/python3.8` |
| `SKIP_SMOKE_TEST` | `0` | 设为 `1` 跳过安装后的 `/health` 检查 |
| `SMOKE_PORT` | `18000` | 冒烟测试端口（只监听 127.0.0.1） |

脚本会校验 wheel、备份旧代码到 `INSTALL_DIR/.backup/<时间>/`、创建或复用 venv、离线安装依赖并做冒烟测试。

### 4. 后台运行（systemd）

`/etc/systemd/system/image-process.service`：

```ini
[Unit]
Description=Image Process Service
After=network.target

[Service]
User=imgproc
WorkingDirectory=/opt/image-process
ExecStart=/opt/image-process/venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
Restart=on-failure
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin imgproc
sudo chown -R imgproc:imgproc /opt/image-process
sudo systemctl daemon-reload && sudo systemctl enable --now image-process
curl --noproxy '*' -s http://127.0.0.1:8000/health   # {"status":"ok","version":"1.0.0"}
```

`--workers` 一般取 CPU 核数。走 Nginx 反向代理时设置 `client_max_body_size 12M;`、`proxy_read_timeout 120s;`。

### 升级与回滚

- 升级：重新打包 → 现场用相同 `INSTALL_DIR` 再执行 `install_offline.sh` → `systemctl restart image-process`。
- 回滚：`cp -r /opt/image-process/.backup/<时间>/. /opt/image-process/` 后重启；依赖有变化时用旧离线包重装。

## 接口

上传限制 10MB，支持 JPEG/PNG/BMP/TIFF/WebP，自动应用 JPEG 的 EXIF 方向。

- `POST /api/v1/process`：返回图片，处理元数据在响应头 `X-Process-Info`（JSON）。
- `POST /api/v1/process/json`：返回 `{"image": "<base64>", "mime_type": ..., "metadata": {...}, "debug_image": ...}`。

请求为 `multipart/form-data`，非法参数值回退为默认值：

| 参数 | 默认 | 说明 |
|------|------|------|
| `file` | 必填 | 图片文件 |
| `detect_document` | `true` | 检测纸张边界并透视矫正 |
| `deskew` | `true` | 去除小角度倾斜 |
| `fix_orientation` | `true` | 自动 90° 方向矫正 |
| `trim_borders` | `true` | 未检测到纸张时裁黑边、裁到内容区域 |
| `manual_rotation` | `0` | 自动处理后再顺时针旋转 0/90/180/270 |
| `output_format` | `JPEG` | `JPEG` 或 `PNG` |
| `quality` | `90` | JPEG 质量 1-100 |
| `debug` | `false` | 仅 JSON 接口：返回标注检测框的调试图 |

主要元数据：`detected_quad`（原图纸张四角，未检测到为 `null`）、`rotation_applied`（顺时针总旋转角）、`deskew_angle`（正数为逆时针）、`orientation_confidence`（0~1，偏低时建议人工复核或传 `manual_rotation`）、`steps`（实际执行的步骤）。

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:8000/api/v1/process \
  -F "file=@scan.jpg" -D - -o result.jpg | grep -i x-process-info
```

错误码：400 图片无法解码，413 超过大小限制，500 内部错误，响应体为 `{"detail": "原因"}`。

## 调参

参数在 `image_processor.py` 文件头部：

| 参数 | 默认 | 调整方向 |
|------|------|------|
| `MIN_EDGE_CONTRAST` | 25 | 白纸放在浅色底板上检测不到纸边时调低（如 15）；表格外框被误当纸边时调高 |
| `MIN_DOC_AREA_RATIO` | 0.2 | 纸张在画面中很小时调低 |
| `DOC_INSET_RATIO` | 0.004 | 结果边缘残留阴影时调大 |
| `VERTICAL_RATIO` | 1.3 | 横排文档被误转 90° 时调大；竖向未被矫正时调小 |
| `FLIP_180_SCORE` | -0.5 | 倒置未被识别时调高（如 -0.3）；正向被误翻时调低 |

已知局限：方向和倒置判断依赖中文标点与行首对齐等规则，纯表格、纯数字页面可能判断错；书本弯曲页面不做曲面展平；纸张与背景颜色接近时检测不到纸边，只做去倾斜和裁边。

## 常见问题

| 现象 | 处理 |
|------|------|
| `架构不匹配` | 用 `TARGET_ARCH=aarch64` 重新打包 |
| `未找到 Python 3.8` / `is not a supported wheel` | 用 `PYTHON_BIN` 指定 3.8 解释器 |
| `wheel 校验失败` | 传输损坏，重新拷贝并先 `sha256sum -c` |
| `/usr/bin/env: 'bash\r'` | `sed -i 's/\r$//' install_offline.sh` |
| `冒烟测试端口 18000 已被占用` | `SMOKE_PORT=18080` 或 `SKIP_SMOKE_TEST=1` |
| `No module named 'app'` | 未在安装目录下启动，检查 systemd 的 `WorkingDirectory` |
| `ImportError: libGL.so.1` | 误装了 `opencv-python`，应为 headless 版 |
| curl 本机返回 502 或代理页面 | 加 `--noproxy '*'` |
| systemd 启动后立即退出 | `journalctl -u image-process -n 50`，常见为目录属主不是 `imgproc` |
