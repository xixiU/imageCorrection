"""
FastAPI 图像处理服务
提供文档扫描、倾斜矫正、方向修正等功能
"""
import base64
import json
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# 导入核心处理模块（由 image_processor.py 提供）
from image_processor import process_image, ProcessOptions, ProcessResult

app = FastAPI(
    title="图像处理服务",
    description="文档扫描、倾斜矫正、方向修正",
    version="1.0.0"
)

# 挂载静态文件目录
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 配置
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB


def parse_bool(value: Optional[str], default: bool = True) -> bool:
    """解析表单布尔值"""
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def parse_int(value: Optional[str], default: int = 0, allowed: Optional[list] = None) -> int:
    """解析表单整数值"""
    if value is None:
        return default
    try:
        val = int(value)
        if allowed and val not in allowed:
            return default
        return val
    except (ValueError, TypeError):
        return default


@app.get("/", response_class=HTMLResponse)
def root():
    """返回前端页面"""
    index_file = static_dir / "index.html"
    if not index_file.exists():
        return HTMLResponse(
            content="<h1>图像处理服务</h1><p>前端页面未部署</p>",
            status_code=404
        )
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/health")
def health_check():
    """健康检查端点"""
    return {"status": "ok", "version": "1.0.0"}


def read_upload(file: UploadFile) -> bytes:
    """读取上传文件，最多读取上限+1字节，避免超大文件整体读入内存"""
    image_data = file.file.read(MAX_UPLOAD_SIZE + 1)
    if len(image_data) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="文件大小超过限制 ({} MB)".format(MAX_UPLOAD_SIZE // 1024 // 1024)
        )
    if len(image_data) == 0:
        raise HTTPException(status_code=400, detail="上传文件为空")
    return image_data


def build_options(
    detect_document: Optional[str],
    deskew: Optional[str],
    fix_orientation: Optional[str],
    trim_borders: Optional[str],
    manual_rotation: Optional[str],
    output_format: Optional[str],
    quality: Optional[str],
    debug: Optional[str] = None,
) -> ProcessOptions:
    """解析表单参数为处理选项，非法值回退为默认值"""
    fmt = (output_format or "JPEG").upper()
    q = parse_int(quality, 90)
    return ProcessOptions(
        detect_document=parse_bool(detect_document, True),
        deskew=parse_bool(deskew, True),
        fix_orientation=parse_bool(fix_orientation, True),
        trim_borders=parse_bool(trim_borders, True),
        manual_rotation=parse_int(manual_rotation, 0, [0, 90, 180, 270]),
        output_format=fmt if fmt in ("JPEG", "JPG", "PNG") else "JPEG",
        quality=q if 1 <= q <= 100 else 90,
        debug=parse_bool(debug, False),
    )


def run_process(image_data: bytes, options: ProcessOptions) -> ProcessResult:
    try:
        return process_image(image_data, options)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="图像处理失败: {}".format(str(e)))
    except Exception as e:
        raise HTTPException(status_code=500, detail="服务器错误: {}".format(str(e)))


@app.post("/api/v1/process")
def process_image_endpoint(
    file: UploadFile = File(...),
    detect_document: Optional[str] = Form(None),
    deskew: Optional[str] = Form(None),
    fix_orientation: Optional[str] = Form(None),
    trim_borders: Optional[str] = Form(None),
    manual_rotation: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
):
    """
    处理图像并返回处理后的图片（CPU密集任务，使用普通def让FastAPI放到线程池）

    响应:
        - Content-Type: image/jpeg 或 image/png
        - Header X-Process-Info: JSON格式的处理元数据
    """
    image_data = read_upload(file)
    options = build_options(detect_document, deskew, fix_orientation, trim_borders,
                            manual_rotation, output_format, quality)
    result = run_process(image_data, options)

    # 返回图片，元数据放在header中
    return Response(
        content=result.image_bytes,
        media_type=result.mime_type,
        headers={"X-Process-Info": json.dumps(result.to_metadata(), ensure_ascii=True)}
    )


@app.post("/api/v1/process/json")
def process_image_json_endpoint(
    file: UploadFile = File(...),
    detect_document: Optional[str] = Form(None),
    deskew: Optional[str] = Form(None),
    fix_orientation: Optional[str] = Form(None),
    trim_borders: Optional[str] = Form(None),
    manual_rotation: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    debug: Optional[str] = Form(None),
):
    """
    处理图像并返回JSON（包含base64编码的图片）（CPU密集任务，使用普通def让FastAPI放到线程池）

    响应:
        {
            "image": "base64编码的图片",
            "metadata": {
                "detected_quad": [[x,y], ...],
                "rotation_applied": 90,
                "deskew_angle": -2.3,
                "original_size": [1920, 1080],
                "processed_size": [1654, 2339],
                "processing_time_ms": 234.5
            },
            "debug_image": "base64编码的调试图片（仅在debug=true时）"
        }
    """
    image_data = read_upload(file)
    options = build_options(detect_document, deskew, fix_orientation, trim_borders,
                            manual_rotation, output_format, quality, debug)
    result = run_process(image_data, options)

    # 构建响应
    response = {
        "image": base64.b64encode(result.image_bytes).decode("utf-8"),
        "mime_type": result.mime_type,
        "metadata": result.to_metadata()
    }

    # debug=true 时附带标注了检测四边形的调试图
    if result.debug_image_bytes is not None:
        response["debug_image"] = base64.b64encode(result.debug_image_bytes).decode("utf-8")

    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
