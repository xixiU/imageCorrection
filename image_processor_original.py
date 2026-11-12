# -*- coding: utf-8 -*-
"""原始工程实现（完全参考 evibhm/ImageCorrection）

完全复制参考工程的算法流程：
1. 图像缩放到900px高度进行处理
2. Canny边缘检测（固定参数：60, 240）
3. 膨胀操作1次（3x3 kernel）
4. 找最大轮廓
5. 凸包+多边形拟合获取四个顶点
6. 透视变换到矩形
7. 自适应阈值二值化（可选）

与参考工程唯一的区别：不使用 scipy，改用纯 OpenCV 实现
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class ProcessOptions:
    """处理选项（与原始工程参数对应）"""
    canny_thresh1: int = 60           # Canny低阈值
    canny_thresh2: int = 240          # Canny高阈值
    dilate_iterations: int = 1        # 膨胀次数
    adaptive_window: int = 9          # 自适应阈值窗口大小（奇数）
    adaptive_beta: float = 0.9        # 自适应阈值系数
    manual_rotation: int = 0          # 手动旋转角度 0/90/180/270
    output_format: str = 'JPEG'       # 输出格式
    quality: int = 95                 # JPEG质量
    apply_adaptive_threshold: bool = False  # 是否应用自适应阈值二值化
    debug: bool = False               # 调试模式


@dataclass
class ProcessResult:
    """处理结果"""
    image_bytes: bytes
    mime_type: str
    detected_quad: Optional[List[List[int]]]
    rotation_applied: int
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    processing_time_ms: float
    steps: List[str] = field(default_factory=list)
    debug_image_bytes: Optional[bytes] = None
    debug_images: dict = field(default_factory=dict)

    def to_metadata(self):
        return {
            'detected_quad': self.detected_quad,
            'rotation_applied': self.rotation_applied,
            'original_size': list(self.original_size),
            'processed_size': list(self.processed_size),
            'processing_time_ms': round(self.processing_time_ms, 1),
            'steps': list(self.steps),
        }


# ---------------- 编解码 ----------------
def decode_image(image_bytes):
    """解码图像"""
    if not image_bytes:
        raise ValueError('图片内容为空')
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('无法解码图片')
    return img


def encode_image(img, fmt='JPEG', quality=95):
    """编码图像"""
    fmt = fmt.upper()
    if fmt in ('JPEG', 'JPG'):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        mime = 'image/jpeg'
    elif fmt == 'PNG':
        ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        mime = 'image/png'
    else:
        raise ValueError('不支持的输出格式: %s' % fmt)
    if not ok:
        raise ValueError('图片编码失败')
    return buf.tobytes(), mime


# ---------------- 原始工程算法实现 ----------------
def resize_img(image, height=900):
    """根据指定高度等比例缩放图像（完全复制原始工程）"""
    h, w = image.shape[:2]
    if h == 0:
        return image
    pro = height / h
    size = (int(w * pro), int(height))
    img = cv2.resize(image, size)
    return img, pro


def get_canny(image, thresh1, thresh2, dilate_iterations):
    """边缘检测（完全复制原始工程）"""
    # 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 高斯滤波
    binary = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=2.0, sigmaY=2.0)

    # Canny边缘检测
    binary = cv2.Canny(binary, thresh1, thresh2, apertureSize=3)

    # 膨胀操作
    kernel = np.ones((3, 3), np.uint8)
    if dilate_iterations > 0:
        binary = cv2.dilate(binary, kernel, iterations=dilate_iterations)

    return binary


def find_max_contour(image):
    """找出面积最大的轮廓（完全复制原始工程）"""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 0.0
    max_contour = []
    for contour in contours:
        current_area = cv2.contourArea(contour)
        if current_area > max_area:
            max_area = current_area
            max_contour = contour
    return max_contour, max_area


def get_box_point(contour):
    """通过凸包和多边形拟合获取四个顶点（完全复制原始工程）"""
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx


def ada_point(box, pro):
    """根据比例调整点集坐标（完全复制原始工程）"""
    box_pro = box.astype(np.float32)
    if pro != 0:
        box_pro = box_pro / pro
    return box_pro.astype(np.float32)


def order_points(pts):
    """对四边形顶点排序（完全复制原始工程）

    顺序：[左上, 右上, 右下, 左下]
    """
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上：x+y最小
    rect[2] = pts[np.argmax(s)]  # 右下：x+y最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上：x-y最小
    rect[3] = pts[np.argmax(diff)]  # 左下：x-y最大
    return rect


def point_distance(a, b):
    """计算两点间欧氏距离（完全复制原始工程）"""
    return int(np.sqrt(np.sum(np.square(a - b))))


def warp_image(image, box):
    """透视变换（完全复制原始工程）"""
    box = np.array(box, dtype='float32')
    if box.shape[0] != 4:
        print("警告: 透视变换需要4个角点，返回原图")
        return image.copy()

    ordered_box = order_points(box)

    # 计算目标矩形的宽度和高度
    widthA = point_distance(ordered_box[0], ordered_box[1])
    widthB = point_distance(ordered_box[3], ordered_box[2])
    maxWidth = max(int(widthA), int(widthB))

    heightA = point_distance(ordered_box[0], ordered_box[3])
    heightB = point_distance(ordered_box[1], ordered_box[2])
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth <= 0 or maxHeight <= 0:
        print("警告: 计算出的变换后图像宽度或高度无效，返回原图")
        return image.copy()

    dst_rect = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype='float32'
    )

    try:
        M = cv2.getPerspectiveTransform(ordered_box, dst_rect)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    except cv2.error as e:
        print(f"透视变换时发生OpenCV错误: {e}")
        return image.copy()

    return warped


def adaptive_threshold_opencv(img, win, beta):
    """自适应阈值（使用纯OpenCV实现，避免scipy依赖）

    原始工程使用 scipy.signal.correlate2d，这里改用 cv2.filter2D 实现相同效果
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        img_gray = img
    else:
        print("错误：adaptive_threshold 的输入图像格式不正确")
        return img

    # 确保窗口大小是奇数
    if win % 2 == 0:
        win = win + 1 if win > 0 else 3

    # 使用均值滤波计算局部均值（等效于 scipy.signal.correlate2d）
    img_gray_float = img_gray.astype(np.float32)
    kernel = np.ones((win, win), np.float32) / (win * win)
    means = cv2.filter2D(img_gray_float, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    # 应用阈值
    binary_img = np.where(img_gray < means * beta, 0, 255)
    return binary_img.astype(np.uint8)


def _rotate90(img, clockwise_deg):
    """90度旋转"""
    clockwise_deg %= 360
    if clockwise_deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if clockwise_deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if clockwise_deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# ---------------- 主处理流程 ----------------
def process_image(image_bytes, options=None):
    """图像处理主函数（完全参考原始工程流程）"""
    if options is None:
        options = ProcessOptions()

    t0 = time.time()

    # 解码图像
    original = decode_image(image_bytes)
    img = original.copy()
    h0, w0 = img.shape[:2]
    steps = []
    quad = None
    debug_images = {}

    print(f"[原始工程] 开始处理: 原始尺寸 {w0}x{h0}")

    # 1. 缩放到900px高度进行处理
    target_height = 900
    img_resized, scaling_ratio = resize_img(img, height=target_height)
    print(f"[原始工程] 缩放图像: {img_resized.shape[1]}x{img_resized.shape[0]}, 缩放比例={scaling_ratio:.4f}")
    steps.append('resize')

    # 2. Canny边缘检测
    canny_result = get_canny(
        img_resized,
        options.canny_thresh1,
        options.canny_thresh2,
        options.dilate_iterations
    )
    print(f"[原始工程] Canny边缘检测完成")
    steps.append('canny')

    # 保存边缘图像用于调试
    if options.debug:
        edges_bgr = cv2.cvtColor(canny_result, cv2.COLOR_GRAY2BGR)
        debug_images['edges_image'] = encode_image(edges_bgr, 'JPEG', 85)[0]

    # 3. 找最大轮廓
    max_contour, max_area = find_max_contour(canny_result)

    if len(max_contour) == 0:
        print(f"[原始工程] 警告：未找到轮廓，返回原图")
        img_final = img
    else:
        print(f"[原始工程] 找到最大轮廓，面积={max_area:.0f}")
        steps.append('find_contour')

        # 4. 获取四边形顶点
        boxes_on_resized = get_box_point(max_contour)

        if boxes_on_resized.shape[0] < 4:
            print(f"[原始工程] 警告：轮廓点不足4个（实际{boxes_on_resized.shape[0]}个），返回原图")
            img_final = img
        else:
            print(f"[原始工程] 获取到{boxes_on_resized.shape[0]}个顶点")
            steps.append('get_quad')

            # 5. 坐标缩放到原始图像
            boxes_on_original = ada_point(boxes_on_resized, scaling_ratio)
            ordered_boxes = order_points(boxes_on_original.astype(np.float32))
            quad = ordered_boxes

            print(f"[原始工程] 四边形顶点（原始尺寸）: {ordered_boxes.tolist()}")

            # 保存轮廓图像用于调试
            if options.debug:
                contour_img = img.copy()
                cv2.drawContours(
                    contour_img,
                    [ordered_boxes.astype(np.int32)],
                    -1,
                    (0, 255, 0),
                    3
                )
                debug_images['contours_image'] = encode_image(contour_img, 'JPEG', 85)[0]

            # 6. 透视变换
            warped = warp_image(img, ordered_boxes)

            # 检查透视变换是否成功
            if warped.size < 100 or np.array_equal(warped, img):
                print(f"[原始工程] 警告：透视变换失败，返回原图")
                img_final = img
            else:
                print(f"[原始工程] 透视变换完成: {warped.shape[1]}x{warped.shape[0]}")
                steps.append('warp')
                img_final = warped

                # 保存透视变换后的图像用于调试
                if options.debug:
                    debug_images['warped_image'] = encode_image(img_final, 'JPEG', 85)[0]

                # 7. 自适应阈值二值化（可选）
                if options.apply_adaptive_threshold:
                    img_final = adaptive_threshold_opencv(
                        img_final,
                        options.adaptive_window,
                        options.adaptive_beta
                    )
                    # 转回BGR（如果是灰度）
                    if len(img_final.shape) == 2:
                        img_final = cv2.cvtColor(img_final, cv2.COLOR_GRAY2BGR)
                    print(f"[原始工程] 自适应阈值二值化完成")
                    steps.append('adaptive_threshold')

    # 8. 手动旋转
    if options.manual_rotation:
        img_final = _rotate90(img_final, options.manual_rotation)
        steps.append('manual_rotation')
        print(f"[原始工程] 手动旋转: {options.manual_rotation}°")

    # 9. 编码输出
    data, mime = encode_image(img_final, options.output_format, options.quality)

    processing_time = (time.time() - t0) * 1000.0
    print(f"[原始工程] 处理完成: 最终尺寸 {img_final.shape[1]}x{img_final.shape[0]}, 耗时 {processing_time:.1f}ms")

    return ProcessResult(
        image_bytes=data,
        mime_type=mime,
        detected_quad=quad.round().astype(int).tolist() if quad is not None else None,
        rotation_applied=options.manual_rotation,
        original_size=(w0, h0),
        processed_size=(img_final.shape[1], img_final.shape[0]),
        processing_time_ms=processing_time,
        steps=steps,
        debug_images=debug_images,
    )


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('用法: python image_processor_original.py 输入图片 输出图片(.jpg/.png)')
        sys.exit(1)

    with open(sys.argv[1], 'rb') as f:
        raw = f.read()

    out_fmt = 'PNG' if sys.argv[2].lower().endswith('.png') else 'JPEG'
    result = process_image(raw, ProcessOptions(output_format=out_fmt, debug=True))

    with open(sys.argv[2], 'wb') as f:
        f.write(result.image_bytes)

    import json
    print(json.dumps(result.to_metadata(), ensure_ascii=False, indent=2))
