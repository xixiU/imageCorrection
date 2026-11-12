# 高拍仪扫描件矫正服务

自动矫正扫描件：纸张检测与透视矫正 → 去倾斜 → 90° 方向矫正 → 裁黑边。纯 OpenCV + numpy，无深度学习模型，Python 3.8。

## 后端版本切换

本项目支持两种处理后端：

1. **current（默认）**：当前优化版本
   - 支持完整的纸张检测、去倾斜、方向矫正、裁边等功能
   - 多尺度边缘检测、动态对比度阈值、笔画方向检测等增强算法
   
2. **original**：原始工程版本（参考 [evibhm/ImageCorrection](https://github.com/evibhm/ImageCorrection)）
   - 完全复制参考工程的算法流程和参数
   - 简单可靠的边缘检测和透视变换

### 切换方法

**方法1：环境变量**
```bash
# 使用原始工程版本
export BACKEND=original
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# 使用当前优化版本（默认）
export BACKEND=current
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**方法2：命令行测试**
```bash
# 测试原始工程版本
python image_processor_original.py input.jpg output_original.jpg

# 测试当前优化版本
python image_processor.py input.jpg output_current.jpg

# 对比两种版本
python test_backends.py input.jpg both
```

### 版本对比

| 特性 | 原始工程版本 | 当前优化版本 |
|------|-------------|-------------|
| 纸张检测 | 单一Canny边缘检测 | 多尺度Canny + Otsu二值化 |
| 对比度阈值 | 固定阈值 | 动态阈值（面积自适应） |
| 去倾斜 | 不支持 | 支持（投影能量法） |
| 方向矫正 | 不支持 | 支持（笔画+标点检测） |
| 倒置检测 | 不支持 | 支持（标点位置分析） |
| 裁边 | 不支持 | 支持（文字区域检测） |
| 处理速度 | ~160ms | ~420ms |
| 输出质量 | 基础矫正 | 全自动多步处理 |

## 快速开始

### 环境要求
- Linux x86_64/aarch64，glibc ≥ 2.17
- Python 3.8
- 端口 8000

### 部署机器：联网安装
```bash
cd /opt/image-process
python3.8 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://<IP>:8000/` 上传测试。

### 打离线包
```bash
bash scripts/build_offline.sh                    # x86_64
TARGET_ARCH=aarch64 bash scripts/build_offline.sh # ARM
```

产物：`dist/image-process-offline.tar.gz` 和 `.sha256`

### 现场：离线安装
```bash
cd /tmp
sha256sum -c image-process-offline.tar.gz.sha256
tar -xzf image-process-offline.tar.gz
cd image-process-offline
sudo bash install_offline.sh
```

后台运行见 `docs/implementation/部署说明.md`（systemd、Nginx、升级）。

## 接口

### POST /api/v1/process/json
返回 JSON（base64 图片 + 元数据）。

**请求**（`multipart/form-data`，上传限制 10MB）：
| 参数 | 默认 | 说明 |
|------|------|------|
| `file` | 必填 | 图片文件（JPEG/PNG/BMP/TIFF/WebP） |
| `detect_document` | `true` | 检测纸张边界并透视矫正 |
| `deskew` | `true` | 去除小角度倾斜 |
| `fix_orientation` | `true` | 自动 90° 方向矫正 |
| `trim_borders` | `true` | 未检测到纸张时裁黑边、裁到内容区域 |
| `manual_rotation` | `0` | 自动处理后再顺时针旋转 0/90/180/270 |
| `output_format` | `JPEG` | `JPEG` 或 `PNG` |
| `quality` | `90` | JPEG 质量 1-100 |
| `debug` | `false` | 额外返回原图标注检测框的调试图 |

**响应**：
```json
{
  "image": "<base64>",
  "mime_type": "image/jpeg",
  "metadata": {
    "detected_quad": [[x,y], ...],
    "rotation_applied": 90,
    "deskew_angle": -2.3,
    "orientation_confidence": 0.62,
    "original_size": [4032, 3024],
    "processed_size": [2480, 3508],
    "processing_time_ms": 850.2,
    "steps": ["document_warp", "deskew", "orientation"]
  },
  "debug_image": "<base64，仅 debug=true 时有>"
}
```

### POST /api/v1/process
直接返回图片，元数据在响应头 `X-Process-Info`（JSON 字符串）。

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/process \
  -F "file=@scan.jpg" -D - -o result.jpg | grep x-process-info
```

### Python 调用
```python
import base64, requests
with open('scan.jpg', 'rb') as f:
    r = requests.post('http://服务器:8000/api/v1/process/json',
                      files={'file': f}, timeout=60)
data = r.json()
open('result.jpg', 'wb').write(base64.b64decode(data['image']))
print(data['metadata'])
```

## 调参

`orientation_confidence` 偏低时，页面可能转错方向，建议传 `manual_rotation` 手动指定。

可调参数在 `image_processor.py` 文件头部：
- `MIN_EDGE_CONTRAST`（默认 25）：白纸放浅色底板上检测不到纸边时调低（如 15）
- `VERTICAL_RATIO`（默认 1.3）：横排文档被误转 90° 时调大
- `FLIP_180_SCORE`（默认 -0.5）：倒置未被识别时调高（如 -0.3）

## 已知局限
- 90°/270° 和倒置靠标点位置、行首对齐等经验规则，纯表格/纯数字页面可能判断错
- 纸张与背景颜色接近时检测不到纸边，只做去倾斜和裁边
- 书本弯曲页面不做曲面展平

详见 `docs/implementation/部署说明.md`。
