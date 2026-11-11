# -*- coding: utf-8 -*-
"""高拍仪扫描件矫正与裁剪核心算法（纯 OpenCV + numpy，兼容 Python 3.8）

处理流程：
1. 解码：cv2.imdecode 会自动应用 JPEG 的 EXIF 方向
2. 纸张检测：边缘、亮度两路找四边形，校验四边形内外色差后做透视矫正
3. 去倾斜：在 ±45°（已透视矫正时 ±5°）内搜索文字投影能量最大的角度
4. 方向矫正：行/列投影判断横排还是竖排（0°/90°），标点位置与行首对齐判断上下
5. 去边：未检测到纸张时裁掉黑边，并裁到内容区域（保留少量页边距）
6. 手动旋转：最后按调用方指定的角度再顺时针旋转

角度约定：
- rotation_applied：顺时针旋转的角度（0/90/180/270），含自动和手动
- deskew_angle：去倾斜旋转角度，正数表示逆时针
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import json

import cv2
import numpy as np

# ---------------- 通用工具（提前定义供配置加载使用）----------------
def _load_config():
    """从 config.json 加载可调参数，覆盖默认值"""
    try:
        from pathlib import Path
        config_path = Path(__file__).parent / 'config.json'
        if not config_path.exists():
            return {}
        with open(config_path, 'r', encoding='utf-8') as f:
            import json
            return json.load(f)
    except Exception:
        return {}

# ---------------- 可调参数 ----------------
# 从 config.json 加载覆盖配置（如果存在）
_config = _load_config()

MAX_PIXELS = _config.get('MAX_PIXELS', 60000000)          # 解码后像素上限，防止超大图耗尽内存
DETECT_MAX_SIDE = _config.get('DETECT_MAX_SIDE', 1000)         # 纸张检测时的工作分辨率
MIN_DOC_AREA_RATIO = _config.get('MIN_DOC_AREA_RATIO', 0.2)       # 纸张至少占画面的比例
MAX_DOC_AREA_RATIO = _config.get('MAX_DOC_AREA_RATIO', 0.97)      # 超过该比例视为纸张铺满画面，不做透视
MIN_EDGE_CONTRAST = _config.get('MIN_EDGE_CONTRAST', 10.0)       # 纸张边缘内外 LAB 色差下限，降低到10.0以减少漏检（默认15.0太严格）
DOC_INSET_RATIO = _config.get('DOC_INSET_RATIO', -0.005)        # 透视前四角向外扩展比例（负值=向外扩展），避免裁掉边缘内容
DESKEW_MAX_SIDE = _config.get('DESKEW_MAX_SIDE', 800)          # 去倾斜计算分辨率
MIN_DESKEW_ANGLE = _config.get('MIN_DESKEW_ANGLE', 10.0)        # 只处理大角度倾斜，避免质量损失（从0.3提高到10.0）
MIN_SKEW_CONFIDENCE = _config.get('MIN_SKEW_CONFIDENCE', 0.05)     # 投影能量峰值相对中位数的提升下限
ORIENT_MAX_SIDE = _config.get('ORIENT_MAX_SIDE', 1600)         # 方向判断计算分辨率
HORIZONTAL_RATIO = _config.get('HORIZONTAL_RATIO', 1.15)        # 行/列投影特征比值 >= 该值判为横排
VERTICAL_RATIO = _config.get('VERTICAL_RATIO', 1.3)           # 列/行投影特征比值 >= 该值判为竖排（保守，避免误转）
MIN_PUNCT_FOR_90 = _config.get('MIN_PUNCT_FOR_90', 4)           # 判断 90°/270° 时标点证据的最少数量
MIN_PUNCT_FOR_180 = _config.get('MIN_PUNCT_FOR_180', 8)          # 判断倒置时标点证据的最少数量
FLIP_180_SCORE = _config.get('FLIP_180_SCORE', -0.5)          # 标点得分低于该值才认为倒置


@dataclass
class ProcessOptions:
    detect_document: bool = True     # 检测纸张边界并透视矫正
    deskew: bool = True              # 去除小角度倾斜
    fix_orientation: bool = True     # 修正 90° 旋转
    detect_upside_down: bool = True  # 修正倒置（证据充分时才翻转）
    text_orientation_correction: bool = False  # 文字方向矫正（降低阈值，更主动旋转）
    trim_borders: bool = True        # 未检测到纸张时裁黑边、裁到内容区域
    manual_rotation: int = 0         # 处理后再顺时针旋转 0/90/180/270
    output_format: str = 'JPEG'      # JPEG / PNG
    quality: int = 90                # JPEG 质量 1-100
    binarize: bool = False           # 自适应阈值二值化（增强文字对比度）
    min_edge_contrast: Optional[int] = None  # 动态调整边缘对比度阈值（10-50，None使用全局配置）
    debug: bool = False              # 生成标注了检测四边形的调试图
    skip_perspective_warp: bool = False  # 跳过透视变换，使用旋转+裁剪（默认False，使用透视变换保证精确裁剪）


@dataclass
class ProcessResult:
    image_bytes: bytes
    mime_type: str
    detected_quad: Optional[List[List[int]]]
    rotation_applied: int
    deskew_angle: float
    orientation_confidence: float
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    processing_time_ms: float
    steps: List[str] = field(default_factory=list)
    debug_image_bytes: Optional[bytes] = None
    debug_images: dict = field(default_factory=dict)  # 多张中间图：edges_image、contours_image等
    flipped_180: bool = False        # 是否进行了 180° 翻转
    flip_confidence: float = 0.0     # 翻转置信度
    irregular_boundary: bool = False # 检测到不规则边界（缺角等）
    corner_repaired: bool = False    # 是否进行了缺角修复

    def to_metadata(self):
        return {
            'detected_quad': self.detected_quad,
            'rotation_applied': self.rotation_applied,
            'deskew_angle': round(self.deskew_angle, 2),
            'orientation_confidence': round(self.orientation_confidence, 2),
            'original_size': list(self.original_size),
            'processed_size': list(self.processed_size),
            'processing_time_ms': round(self.processing_time_ms, 1),
            'steps': list(self.steps),
            'flipped_180': self.flipped_180,
            'flip_confidence': round(self.flip_confidence, 2),
            'irregular_boundary': self.irregular_boundary,
            'corner_repaired': self.corner_repaired,
        }


# ---------------- 编解码 ----------------
def decode_image(image_bytes):
    if not image_bytes:
        raise ValueError('图片内容为空')
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('无法解码图片，支持 JPEG/PNG/BMP/TIFF/WebP')
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS:
        raise ValueError('图片尺寸过大: %dx%d' % (w, h))
    return img


def encode_image(img, fmt='JPEG', quality=95):
    """编码图像为字节流。默认 JPEG 质量 95（高质量）"""
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


# ---------------- 通用工具 ----------------
def _resize_max(img, max_side):
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img, scale


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _text_mask(gray, remove_lines=True):
    """文字墨迹掩码（0/1）。大面积深色区域只会留下轮廓，不会整体变成墨迹。"""
    h, w = gray.shape
    block = max(15, (min(h, w) // 40) | 1)
    mask = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, block, 15)
    if remove_lines:
        # 去掉表格线、黑边等长直线，避免干扰文字方向判断
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 15), 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 15)))
        lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kh) | cv2.morphologyEx(mask, cv2.MORPH_OPEN, kv)
        mask[lines > 0] = 0
    return mask


def _paper_color(img):
    small, _ = _resize_max(img, 400)
    gray = _to_gray(small)
    px = small[gray >= np.percentile(gray, 60)]
    if len(px) == 0:
        return (255, 255, 255)
    return tuple(int(v) for v in np.median(px, axis=0))


def _rotate_image(img, angle, fill, expand):
    """按角度旋转（正数逆时针）。expand=True 时扩大画布，避免内容被裁掉。"""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    nw, nh = w, h
    if expand:
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw = int(np.ceil(h * sin + w * cos))
        nh = int(np.ceil(h * cos + w * sin))
        m[0, 2] += nw / 2.0 - w / 2.0
        m[1, 2] += nh / 2.0 - h / 2.0
    # 使用最高质量插值（LANCZOS4）避免质量损失
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def _rotate90(img, clockwise_deg):
    clockwise_deg %= 360
    if clockwise_deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if clockwise_deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if clockwise_deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# ---------------- 纸张检测与透视矫正 ----------------
def _order_quad(pts):
    """按 左上、右上、右下、左下 排序。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # 图像坐标系 y 向下，角度递增即顺时针
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def _is_irregular_quad(q, area_ratio):
    """综合判断四边形是否不规则（缺角、变形等）

    检查项：
    1. 角度异常（< 60° 或 > 120°）
    2. 面积比过小（< 50%，可能是缺角）
    3. 长宽比异常（偏离 A4 比例太多）
    4. 对边长度差异过大（> 30%）

    返回：(是否不规则, 原因描述)
    """
    # 1. 角度检查
    for i in range(4):
        v1 = q[i - 1] - q[i]
        v2 = q[(i + 1) % 4] - q[i]
        cos_v = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        ang = np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0)))
        if ang < 60 or ang > 120:
            return True, f"角度异常 {ang:.1f}°"

    # 2. 面积比检查（关键！缺角纸张面积明显偏小）
    if area_ratio < 0.5:
        return True, f"面积比过小 {area_ratio:.1%}"

    # 3. 长宽比检查
    # 计算四边形的宽度和高度（使用边界框）
    xs, ys = q[:, 0], q[:, 1]
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    aspect = max(width, height) / (min(width, height) + 1e-6)
    # 放宽范围：0.9-2.2，覆盖接近正方形到竖向 A4 的所有合理比例
    if aspect < 0.9 or aspect > 2.2:
        return True, f"长宽比异常 {aspect:.2f}"

    # 4. 对边长度差异检查
    edge_lengths = []
    for i in range(4):
        edge = np.linalg.norm(q[(i + 1) % 4] - q[i])
        edge_lengths.append(edge)

    # 对边：0-2, 1-3
    diff_02 = abs(edge_lengths[0] - edge_lengths[2]) / max(edge_lengths[0], edge_lengths[2])
    diff_13 = abs(edge_lengths[1] - edge_lengths[3]) / max(edge_lengths[1], edge_lengths[3])

    if diff_02 > 0.3 or diff_13 > 0.3:
        return True, f"对边差异过大 {max(diff_02, diff_13):.1%}"

    return False, None


def _quad_angles_ok(q, strict_check=False):
    """检查四边形角度是否合理。

    strict_check=True: 更严格检查（60-120°），用于识别不规则边界
    strict_check=False: 原有检查（50-130°），用于常规过滤
    """
    min_angle = 60 if strict_check else 50
    max_angle = 120 if strict_check else 130

    angles = []
    for i in range(4):
        v1 = q[i - 1] - q[i]
        v2 = q[(i + 1) % 4] - q[i]
        cos_v = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        ang = np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0)))
        angles.append(ang)
        if ang < min_angle or ang > max_angle:
            if strict_check:
                print(f"[DEBUG] 检测到不规则角度: {ang:.1f}° (期望 {min_angle}-{max_angle}°)")
            return False
    return True


def _contour_to_quad(hull):
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(hull)
    rect_area = rect[1][0] * rect[1][1]
    if rect_area > 0 and cv2.contourArea(hull) / rect_area > 0.9:
        return cv2.boxPoints(rect).astype(np.float32)
    return None


def _edge_contrast(lab, quad):
    """四边形外侧窄带与内侧窄带的 LAB 中位色差。外侧像素不足时返回 0（无法确认）。"""
    h, w = lab.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    band = max(3, int(min(h, w) * 0.015))
    kernel = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
    outer = cv2.subtract(cv2.dilate(mask, kernel), mask)
    inner = cv2.subtract(mask, cv2.erode(mask, kernel))
    if cv2.countNonZero(outer) < 200 or cv2.countNonZero(inner) < 200:
        return 0.0
    mo = np.median(lab[outer > 0], axis=0)
    mi = np.median(lab[inner > 0], axis=0)
    return float(np.linalg.norm(mo - mi))


def _find_document_quad(img_small, min_edge_contrast=None):
    """在缩小图上找纸张四边形，找不到返回 None。

    优化策略：
    1. 简化边缘检测，参考 evibhm/ImageCorrection 的简洁策略
    2. 优先找最大轮廓，减少过滤
    3. 降低色差阈值，减少漏检

    参数：
        min_edge_contrast: 可选，动态覆盖全局 MIN_EDGE_CONTRAST
    """
    # 使用传入的阈值或全局配置
    edge_contrast_threshold = min_edge_contrast if min_edge_contrast is not None else MIN_EDGE_CONTRAST

    gray = cv2.GaussianBlur(_to_gray(img_small), (5, 5), 0)
    h, w = gray.shape
    area = float(h * w)
    contours = []
    debug_edges = None
    debug_contours = None

    # 优化1：简化 Canny 参数，使用更稳定的固定阈值
    # 参考 evibhm/ImageCorrection: thresh1=60, thresh2=240 效果很好
    med = float(np.median(gray))
    for lo, hi in ((60, 180), (0.66 * med, 1.33 * med)):
        edges = cv2.Canny(gray, lo, hi)
        if debug_edges is None:
            debug_edges = edges.copy()  # 保存第一个edges用于调试
        # 优化2：减少膨胀次数，避免边缘过度扩张
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cs)

    # 纸张通常比背景亮：Otsu 二值化后取外轮廓
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(5, int(min(h, w) * 0.02)) | 1
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.extend(cs)

    # 绘制轮廓用于调试
    debug_contours = cv2.cvtColor(img_small.copy(), cv2.COLOR_BGR2RGB)
    cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)

    print(f"[DEBUG] 文档检测: 总轮廓数={len(contours)}, 边缘阈值={edge_contrast_threshold}, 图像尺寸={w}x{h}")

    lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2LAB).astype(np.float32)
    best, best_ratio = None, 0.0
    filtered_count = {'too_small': 0, 'no_quad': 0, 'area_mismatch': 0, 'not_convex': 0, 'irregular': 0, 'low_contrast': 0}

    # 优化3：按面积排序轮廓，优先处理大轮廓（参考 evibhm/ImageCorrection 找最大轮廓的策略）
    contours_with_area = [(cnt, cv2.contourArea(cv2.convexHull(cnt))) for cnt in contours]
    contours_sorted = sorted(contours_with_area, key=lambda x: x[1], reverse=True)

    for cnt, hull_area in contours_sorted:
        if hull_area < MIN_DOC_AREA_RATIO * area:
            filtered_count['too_small'] += 1
            continue

        hull = cv2.convexHull(cnt)
        quad = _contour_to_quad(hull)
        if quad is None:
            filtered_count['no_quad'] += 1
            continue
        quad = _order_quad(quad)
        ratio = cv2.contourArea(quad) / area

        # 只接受比当前最优更大的候选，减少色差计算次数
        if ratio < MIN_DOC_AREA_RATIO or ratio > MAX_DOC_AREA_RATIO or ratio <= best_ratio:
            filtered_count['area_mismatch'] += 1
            continue
        if not cv2.isContourConvex(quad.reshape(-1, 1, 2)) or not _quad_angles_ok(quad):
            filtered_count['not_convex'] += 1
            continue

        # 优化4：放宽不规则检查，只在极端情况下过滤
        is_irregular, reason = _is_irregular_quad(quad, ratio)
        if is_irregular and ratio < 0.4:  # 只有面积比很小且不规则时才过滤
            filtered_count['irregular'] += 1
            print(f"[DEBUG] 过滤不规则四边形: 面积比={ratio:.2%}, 原因={reason}")
            continue

        # 优化5：色差检查更宽松，或者对大面积轮廓跳过色差检查
        contrast = _edge_contrast(lab, quad)
        if contrast < edge_contrast_threshold and ratio < 0.7:  # 大面积轮廓（>70%）跳过色差检查
            filtered_count['low_contrast'] += 1
            print(f"[DEBUG] 过滤轮廓: 面积比={ratio:.2%}, 对比度={contrast:.1f} < {edge_contrast_threshold}")
            continue
        best, best_ratio = quad, ratio
        print(f"[DEBUG] 找到候选四边形: 面积比={ratio:.2%}, 对比度={contrast:.1f}")
        break  # 找到第一个有效的大轮廓就停止

    print(f"[DEBUG] 过滤统计: {filtered_count}, 最终检测到={'有效四边形' if best is not None else '无'}")
    if best is not None:
        print(f"[DEBUG] 最佳四边形: 面积比={best_ratio:.2%}, 坐标={best.tolist()}")

    # 如果没有找到有效四边形，尝试找出被过滤的最大候选用于修复
    # 优先找被对比度过滤的大面积轮廓（可能是低对比度纸张）
    has_irregular = filtered_count.get('irregular', 0) > 0
    irregular_candidate = None

    if best is None:
        print(f"[DEBUG] 未找到有效四边形，尝试找出被过滤的最大候选用于修复")
        max_ratio = 0.0

        for cnt in contours:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area < MIN_DOC_AREA_RATIO * area:
                continue
            quad = _contour_to_quad(hull)
            if quad is None:
                continue
            quad = _order_quad(quad)
            ratio = cv2.contourArea(quad) / area
            if ratio < MIN_DOC_AREA_RATIO or ratio > MAX_DOC_AREA_RATIO:
                continue

            # 放宽所有检查：只要凸多边形和面积够大
            if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
                continue

            # 找出面积最大的候选（不检查角度、对比度、不规则性）
            if ratio > max_ratio:
                irregular_candidate = quad
                max_ratio = ratio
                has_irregular = True  # 标记为不规则，需要修复

        if irregular_candidate is not None:
            print(f"[DEBUG] 找到被过滤的候选，面积比={max_ratio:.2%}，将尝试修复")

    return best, debug_edges, debug_contours, has_irregular, irregular_candidate


def _fill_black_to_white(img):
    """将图像中的黑色填充区域（透视变换产生）改为白色"""
    # 检测接近黑色的区域（阈值设为50，覆盖深灰色）
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 50
    # 将黑色区域填充为白色
    img[mask] = [255, 255, 255]
    return img


def _line_intersection(line1, line2):
    """计算两条直线的交点

    line1, line2: 每条线由两个点定义 [[x1,y1], [x2,y2]]
    返回：交点 [x, y] 或 None（平行线）
    """
    p1, p2 = line1
    p3, p4 = line2

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None  # 平行线

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return np.array([x, y])


def _repair_irregular_quad(irregular_quad):
    """通过对边延长线交点补齐不规则四边形为完整矩形

    参数：
        irregular_quad: 不规则四边形 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                       顺序：左上、右上、右下、左下

    返回：
        完整矩形的四个顶点（补齐后）
    """
    q = irregular_quad.copy()

    # 四条边：上、右、下、左
    # 通过对边延长线交点推算缺失的角

    # 计算每条边的长度，找出明显较短的边（可能缺角）
    edge_lengths = []
    for i in range(4):
        length = np.linalg.norm(q[(i + 1) % 4] - q[i])
        edge_lengths.append(length)

    avg_length = np.mean(edge_lengths)

    # 如果某条边明显比平均长度短很多（< 70%），可能是缺角导致
    # 这里采用更通用的方法：对所有对边求延长线交点，选择最合理的矩形

    # 定义四条边：0-1(上), 1-2(右), 2-3(下), 3-0(左)
    edges = [(q[i], q[(i + 1) % 4]) for i in range(4)]

    # 计算对边延长线交点
    # 上边(0-1) 和 下边(2-3) 延长
    # 左边(3-0) 和 右边(1-2) 延长

    # 尝试修复：找出哪些顶点可能不准确，用延长线交点替换
    # 简化策略：假设对边应该平行，计算完整矩形的四个角

    # 使用最小外接矩形作为参考
    rect = cv2.minAreaRect(q.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = _order_quad(box)

    print(f"[DEBUG] 不规则四边形修复: 原始顶点={q.tolist()}, 补齐后={box.tolist()}")

    return box.astype(np.float32)


def _inset_quad(quad, ratio):
    """向内收缩或向外扩展四边形

    ratio > 0: 向内收缩（用于去除纸边阴影）
    ratio < 0: 向外扩展（用于避免裁掉边缘内容）
    ratio = 0: 不变
    """
    c = quad.mean(axis=0)
    return (quad + (c - quad) * ratio).astype(np.float32)


def _expand_quad(quad, img_shape, expand_ratio=0.05):
    """向外扩展四边形边界，避免裁掉边缘内容（如标题）

    expand_ratio: 向外扩展的比例（相对于四边形尺寸）
    顶部扩展更多（2倍），防止标题被裁
    """
    h, w = img_shape[:2]
    tl, tr, br, bl = quad

    # 计算四边形的中心和尺寸
    center = quad.mean(axis=0)
    quad_width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    quad_height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

    # 顶部扩展 10%（标题通常在顶部），其他方向 5%
    expand_top = expand_ratio * 2.0
    expand_others = expand_ratio

    # 向外扩展每个顶点
    expanded = []
    for i, pt in enumerate(quad):
        direction = pt - center
        # 顶部两个点（tl, tr）扩展更多
        if i < 2:  # tl, tr
            expanded_pt = pt + direction * expand_top
        else:
            expanded_pt = pt + direction * expand_others
        # 限制在图像范围内
        expanded_pt[0] = np.clip(expanded_pt[0], 0, w - 1)
        expanded_pt[1] = np.clip(expanded_pt[1], 0, h - 1)
        expanded.append(expanded_pt)

    return np.array(expanded, dtype=np.float32)


def _warp_quad(img, quad):
    """透视变换到矩形，保持原始分辨率避免模糊

    优化：参考 evibhm/ImageCorrection 的精确实现
    - 基于实际边长计算目标尺寸（不缩放）
    - 使用双线性插值（默认），平衡质量和性能
    """
    h_orig, w_orig = img.shape[:2]
    tl, tr, br, bl = quad

    # 计算实际边长（参考 evibhm/ImageCorrection 的 pointDistance 逻辑）
    widthA = int(np.linalg.norm(tr - tl))
    widthB = int(np.linalg.norm(br - bl))
    maxWidth = max(widthA, widthB)

    heightA = int(np.linalg.norm(bl - tl))
    heightB = int(np.linalg.norm(br - tr))
    maxHeight = max(heightA, heightB)

    wd, ht = max(maxWidth, 1), max(maxHeight, 1)

    print(f"[DEBUG] 透视变换：输入图像 {w_orig}x{h_orig}, 目标尺寸 {wd}x{ht} (基于实际边长)")

    dst = np.array([[0, 0], [wd - 1, 0], [wd - 1, ht - 1], [0, ht - 1]], np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)

    # 使用双线性插值（cv2.INTER_LINEAR，OpenCV 默认），白色边界填充
    # 参考工程使用默认插值，效果很好，不需要 LANCZOS4
    result = cv2.warpPerspective(img, m, (wd, ht), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    print(f"[DEBUG] 透视变换完成：输出图像 {result.shape[1]}x{result.shape[0]}, 插值=INTER_LINEAR")
    return result


# ---------------- 去倾斜 ----------------
def _rotated_energy(mask, angle):
    """旋转后行投影与列投影的平方和。文字行对齐坐标轴时最大，且对 90° 旋转不敏感。"""
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw = int(h * sin + w * cos) + 1
    nh = int(h * cos + w * sin) + 1
    m[0, 2] += nw / 2.0 - w / 2.0
    m[1, 2] += nh / 2.0 - h / 2.0
    r = cv2.warpAffine(mask, m, (nw, nh), flags=cv2.INTER_NEAREST)
    rows = r.sum(axis=1, dtype=np.float64)
    cols = r.sum(axis=0, dtype=np.float64)
    return float((rows * rows).sum() + (cols * cols).sum())


def _estimate_skew(gray, max_angle):
    """返回 (角度, 置信度)，角度正数表示需逆时针旋转。"""
    small, _ = _resize_max(gray, DESKEW_MAX_SIDE)
    mask = _text_mask(small)
    ink = float(mask.mean())
    if ink < 0.002 or ink > 0.4:
        return 0.0, 0.0
    step = 1.0 if max_angle > 10 else 0.5
    coarse = np.arange(-max_angle, max_angle + 1e-6, step)
    scores = np.array([_rotated_energy(mask, a) for a in coarse])
    i = int(np.argmax(scores))
    conf = float(scores[i] / (np.median(scores) + 1e-6) - 1.0)
    fine = np.arange(coarse[i] - step, coarse[i] + step + 1e-6, 0.1)
    fine_scores = [_rotated_energy(mask, a) for a in fine]
    return float(fine[int(np.argmax(fine_scores))]), conf


# ---------------- 方向判断 ----------------
def _profile_feature(profile):
    """投影的变异系数 + 2×空白占比。文字行方向的投影起伏大、空白多。"""
    p = np.convolve(profile.astype(np.float64), np.ones(3) / 3.0, mode='same')
    total = p.sum()
    if total <= 0:
        return 0.0
    # 用墨迹累计量的 0.5%~99.5% 截取文字范围，忽略边缘零星噪点
    cum = np.cumsum(p) / total
    lo = int(np.searchsorted(cum, 0.005))
    hi = int(np.searchsorted(cum, 0.995)) + 1
    p = p[lo:hi]
    if len(p) < 20 or p.mean() <= 0:
        return 0.0
    cv_val = p.std() / p.mean()
    gap = float((p <= 0.08 * np.percentile(p, 95)).mean())
    return float(cv_val + 2.0 * gap)


def _text_direction(mask):
    """返回 ('h' 横排 / 'v' 竖排 / None 不确定, 行列特征比值)。"""
    fr = _profile_feature(mask.sum(axis=1))
    fc = _profile_feature(mask.sum(axis=0))
    if fr <= 0 or fc <= 0:
        return None, 1.0
    ratio = fr / fc
    if ratio >= HORIZONTAL_RATIO:
        return 'h', ratio
    if ratio <= 1.0 / VERTICAL_RATIO:
        return 'v', ratio
    return None, ratio


def _upright_score(mask):
    """横排文字的正立证据。

    返回 (标点得分, 标点数量, 对齐得分)：
    - 标点得分：中文 。，、 等位于字格下方，独立成列的小连通块在行下部为正、上部为负
    - 对齐得分：行首比行尾整齐为正（段末短行造成行尾参差）
    """
    rows = mask.sum(axis=1).astype(np.float64)
    if rows.max() <= 0:
        return 0.0, 0, 0.0
    on = (rows > 0.03 * np.percentile(rows[rows > 0], 95)).astype(np.int8)
    d = np.diff(np.concatenate([[0], on, [0]]))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    heights = ends - starts
    valid = heights >= 6
    if not valid.any():
        return 0.0, 0, 0.0
    s = float(np.median(heights[valid]))
    max_mark = 0.4 * s
    min_area = max(3, int(0.004 * s * s))

    top = bottom = 0
    lefts, rights = [], []
    for y0, y1 in zip(starts, ends):
        bh = y1 - y0
        if bh < 0.5 * s or bh > 1.8 * s:
            continue
        band = mask[y0:y1]
        col_ink = band.sum(axis=0)
        cols = np.nonzero(col_ink)[0]
        if len(cols) == 0:
            continue
        lefts.append(cols[0])
        rights.append(cols[-1])
        n, _, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if area < min_area or cw > max_mark or ch > max_mark:
                continue
            # 同一列范围内还有其它墨迹说明是字的一部分（如 宀 的点），不是标点
            if col_ink[x:x + cw].sum() - area > 0.15 * area:
                continue
            pos = (y + ch / 2.0) / float(bh)
            if pos > 0.62:
                bottom += 1
            elif pos < 0.38:
                top += 1

    n_punct = top + bottom
    punct = (bottom - top) / float(n_punct) if n_punct else 0.0
    align = 0.0
    if len(lefts) >= 5:
        lefts_a, rights_a = np.array(lefts), np.array(rights)
        left_ok = np.mean(np.abs(lefts_a - np.percentile(lefts_a, 10)) < 0.5 * s)
        right_ok = np.mean(np.abs(np.percentile(rights_a, 90) - rights_a) < 0.5 * s)
        align = float(left_ok - right_ok)
    return punct, n_punct, align


def _detect_orientation(img, detect_upside_down, aggressive=False):
    """返回 (需顺时针旋转的角度, 置信度 0~1, 翻转置信度)。

    改进置信度计算：综合多个指标提高准确性。
    aggressive: 激进模式，降低所有方向判断的阈值
    """
    small, _ = _resize_max(_to_gray(img), ORIENT_MAX_SIDE)
    mask = _text_mask(small)
    ink = float(mask.mean())
    if ink < 0.002 or ink > 0.4:
        return 0, 0.0, 0.0
    direction, ratio = _text_direction(mask)

    if direction == 'v':
        # 文字行是竖的：先按顺时针转 90° 评估，得分为负说明应逆时针转
        punct, n, align = _upright_score(cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE))
        total = (punct if n >= MIN_PUNCT_FOR_90 else 0.0) + 0.5 * align
        # 置信度综合：标点数量 + 对齐得分 + 方向特征比值
        confidence = min(1.0, abs(total) + 0.1 * min(n / float(MIN_PUNCT_FOR_90), 1.0) + 0.2 * (ratio - 1.0))

        # 激进模式：降低标点数量要求
        if aggressive and n >= max(2, MIN_PUNCT_FOR_90 // 2):
            return (90 if total >= 0 else 270), confidence, 0.0

        return (90 if total >= 0 else 270), confidence, 0.0

    if direction == 'h':
        punct, n, align = _upright_score(mask)
        flip_confidence = 0.0

        if detect_upside_down:
            # 计算倒置置信度
            if n >= MIN_PUNCT_FOR_180:
                flip_confidence = min(1.0, -punct + 0.1 * (n / float(MIN_PUNCT_FOR_180)))

            # 激进模式：大幅降低倒置检测要求
            if aggressive:
                if n >= max(4, MIN_PUNCT_FOR_180 // 2) and punct <= -0.3:
                    return 180, flip_confidence, flip_confidence
            else:
                # 保守模式：原有逻辑
                if n >= MIN_PUNCT_FOR_180 and punct <= FLIP_180_SCORE and align <= 0.1:
                    return 180, flip_confidence, flip_confidence

        # 横排正立的置信度：方向特征比值
        return 0, min(1.0, (ratio - 1.0) * 0.8), flip_confidence

    return 0, 0.0, 0.0


# ---------------- 去边与裁剪 ----------------
def _leading_count(flags):
    nz = np.nonzero(~flags)[0]
    return int(nz[0]) if len(nz) else len(flags)


def _trim_dark_borders(img):
    gray = _to_gray(img)
    h, w = gray.shape
    dark = gray < min(90.0, float(np.percentile(gray, 90)) * 0.5)
    rf, cf = dark.mean(axis=1), dark.mean(axis=0)
    top = _leading_count(rf[:h // 4] > 0.5)
    bottom = _leading_count(rf[::-1][:h // 4] > 0.5)
    left = _leading_count(cf[:w // 4] > 0.5)
    right = _leading_count(cf[::-1][:w // 4] > 0.5)
    if top + bottom >= h - 10 or left + right >= w - 10:
        return img
    return img[top:h - bottom, left:w - right]


def _crop_to_content(img):
    """裁到内容区域并保留 2% 页边距；内容过小时不裁。

    增强版：检测失败时，基于亮度/色差主动找桌面边界。
    """
    h, w = img.shape[:2]
    small, scale = _resize_max(_to_gray(img), 1000)
    mask = _text_mask(small, remove_lines=False)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    sh, sw = mask.shape

    # 降低阈值，避免稀疏区域被忽略（表格右侧、下方）
    # 原阈值 0.003 太高，改为 0.0015
    row_threshold = max(1, int(0.0015 * sw))
    col_threshold = max(1, int(0.0015 * sh))
    ys = np.nonzero(mask.sum(axis=1) > row_threshold)[0]
    xs = np.nonzero(mask.sum(axis=0) > col_threshold)[0]

    print(f"[DEBUG] 文字裁剪检测: ys={len(ys)} rows, xs={len(xs)} cols, threshold=({row_threshold}, {col_threshold})")

    # 如果文字墨迹检测失败，尝试基于亮度差异找边界
    if len(ys) == 0 or len(xs) == 0:
        print(f"[DEBUG] 文字墨迹检测失败，尝试梯度检测")
        # 使用边缘检测找纸张与桌面的分界
        small_color, _ = _resize_max(img, 1000)
        lab = cv2.cvtColor(small_color, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = lab[:, :, 0]  # L通道（亮度）

        # 计算梯度
        grad_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
        grad_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5))

        # 从边缘向内扫描，找到第一个明显的边界
        threshold = np.percentile(grad_y, 85)
        y_proj = grad_y.max(axis=1)
        y_edges = np.nonzero(y_proj > threshold)[0]

        threshold = np.percentile(grad_x, 85)
        x_proj = grad_x.max(axis=0)
        x_edges = np.nonzero(x_proj > threshold)[0]

        if len(y_edges) > 0 and len(x_edges) > 0:
            ys = y_edges
            xs = x_edges
            print(f"[DEBUG] 梯度检测成功: ys={len(ys)} rows, xs={len(xs)} cols")
        else:
            print(f"[DEBUG] 梯度检测也失败，保持原图不裁剪")
            return img

    margin = int(max(10, 0.02 * max(h, w)))
    y0 = max(0, int(ys[0] / scale) - margin)
    y1 = min(h, int((ys[-1] + 1) / scale) + margin)
    x0 = max(0, int(xs[0] / scale) - margin)
    x1 = min(w, int((xs[-1] + 1) / scale) + margin)

    crop_area = (y1 - y0) * (x1 - x0)
    original_area = h * w
    crop_ratio = crop_area / original_area

    print(f"[DEBUG] 裁剪区域: ({x0},{y0}) -> ({x1},{y1}), 原始: {w}x{h}, 裁剪后: {x1-x0}x{y1-y0}, 保留比例: {crop_ratio:.2%}")

    # 如果裁剪后面积太小（< 10%），可能是误判，保持原图
    if crop_ratio < 0.1:
        print(f"[DEBUG] 裁剪区域过小（{crop_ratio:.2%}），保持原图")
        return img

    return img[y0:y1, x0:x1]


def _debug_image(img, quad):
    vis, s = _resize_max(img, 1200)
    vis = vis.copy()
    thick = max(2, int(max(vis.shape[:2]) / 300))
    if quad is not None:
        pts = (quad * s).astype(np.int32)
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 200, 0), thick)
        for p in pts:
            cv2.circle(vis, (int(p[0]), int(p[1])), thick * 3, (0, 0, 255), -1)
    else:
        cv2.putText(vis, 'no document edge', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), thick)
    return encode_image(vis, 'JPEG', 85)[0]


def _apply_binarization(img):
    """自适应阈值二值化，增强文字对比度（适合扫描文档）"""
    gray = _to_gray(img)
    # 使用自适应阈值二值化
    h, w = gray.shape
    block_size = max(11, (min(h, w) // 30) | 1)  # 块大小必须是奇数
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, 10
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ---------------- 主入口 ----------------
def _validate_options(o):
    if o.manual_rotation not in (0, 90, 180, 270):
        raise ValueError('manual_rotation 只能是 0/90/180/270')
    if o.output_format.upper() not in ('JPEG', 'JPG', 'PNG'):
        raise ValueError('output_format 只能是 JPEG 或 PNG')
    if not 1 <= int(o.quality) <= 100:
        raise ValueError('quality 取值 1-100')


def process_image(image_bytes, options=None):
    """矫正并裁剪扫描件。图片无法解码或参数非法时抛出 ValueError。"""
    if options is None:
        options = ProcessOptions()
    _validate_options(options)
    t0 = time.time()

    original = decode_image(image_bytes)
    img = original
    h0, w0 = img.shape[:2]
    steps = []
    quad = None
    debug_images = {}
    irregular_boundary = False  # 是否检测到不规则边界
    corner_repaired = False      # 是否进行了缺角修复

    # 【关键修复】先进行纸张检测和透视变换，保持原始分辨率
    # 方向检测移到透视变换之后，避免提前旋转导致分辨率损失
    if options.detect_document and min(h0, w0) >= 200:
        small, scale = _resize_max(img, DETECT_MAX_SIDE)
        q, debug_edges, debug_contours, has_irregular, irregular_candidate = _find_document_quad(small, options.min_edge_contrast)

        # 检测到纸张边界的处理
        if q is not None or (has_irregular and irregular_candidate is not None):
            # 使用正常候选或不规则候选
            detected_quad = q if q is not None else irregular_candidate
            quad_scaled = detected_quad / scale

            if options.skip_perspective_warp:
                # 跳过透视变换，使用旋转最小外接矩形 + 裁剪（保证文字清晰）
                print(f"[DEBUG] 跳过透视变换，使用旋转矩形裁剪（保证文字清晰）")
                rect = cv2.minAreaRect(quad_scaled.astype(np.float32))
                angle_rect = rect[2]
                # minAreaRect 返回角度范围 [-90, 0)，调整到合理范围
                if angle_rect < -45:
                    angle_rect += 90

                print(f"[DEBUG] 旋转矩形角度: {angle_rect:.2f}°")

                # 旋转原图到正向
                if abs(angle_rect) > 0.5:
                    img = _rotate_image(img, angle_rect, _paper_color(img), expand=True)
                    print(f"[DEBUG] 旋转图像: {angle_rect:.2f}°, 尺寸: {img.shape[1]}x{img.shape[0]}")

                # 计算旋转后的矩形边界进行裁剪
                box = cv2.boxPoints(rect)
                box = box.astype(int)  # 修复：numpy 新版本使用 int 替代 int0
                xs, ys = box[:, 0], box[:, 1]
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()

                # 确保在图像范围内
                h, w = img.shape[:2]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                # 裁剪矩形区域
                img = img[y_min:y_max, x_min:x_max]
                print(f"[DEBUG] 裁剪矩形区域完成, 输出尺寸: {img.shape[1]}x{img.shape[0]}")
                steps.append('rotate_crop')
                quad = quad_scaled  # 保存检测到的四边形用于返回

            else:
                # 优化：使用透视变换（默认，精确裁剪边界）
                # 关键改进：不做收缩或向外微扩展，避免裁掉边缘内容
                if has_irregular and irregular_candidate is not None:
                    print(f"[DEBUG] 检测到不规则边界，尝试补齐缺失角")
                    q_repaired = _repair_irregular_quad(irregular_candidate)
                    quad = _inset_quad(q_repaired / scale, DOC_INSET_RATIO)
                    img = _warp_quad(img, quad)
                    img = _fill_black_to_white(img)
                    corner_repaired = True
                    steps.append('corner_repair')
                    print(f"[DEBUG] 缺角修复完成，透视矫正完成")
                else:
                    # 正常四边形，直接透视矫正（不收缩，微向外扩展）
                    quad = _inset_quad(quad_scaled, DOC_INSET_RATIO)
                    img = _warp_quad(img, quad)
                    img = _fill_black_to_white(img)
                    steps.append('document_warp')
                    print(f"[DEBUG] 检测到纸张四边形，透视矫正完成（DOC_INSET_RATIO={DOC_INSET_RATIO}）")
        else:
            print(f"[DEBUG] 未检测到纸张边界 (min_edge_contrast={options.min_edge_contrast or MIN_EDGE_CONTRAST})")

        # 无论 debug 是否开启，都保存中间图（前端需要）
        if debug_edges is not None:
            try:
                edges_bgr = cv2.cvtColor(debug_edges, cv2.COLOR_GRAY2BGR)
                edges_encoded = encode_image(edges_bgr, 'JPEG', 85)[0]
                debug_images['edges_image'] = edges_encoded
                print(f"[DEBUG] edges_image 编码成功，大小: {len(edges_encoded)} bytes")
            except Exception as e:
                print(f"[ERROR] edges_image 编码失败: {e}")
        else:
            print(f"[DEBUG] debug_edges is None")

        if debug_contours is not None:
            try:
                contours_encoded = encode_image(debug_contours, 'JPEG', 85)[0]
                debug_images['contours_image'] = contours_encoded
                print(f"[DEBUG] contours_image 编码成功，大小: {len(contours_encoded)} bytes")
            except Exception as e:
                print(f"[ERROR] contours_image 编码失败: {e}")
        else:
            print(f"[DEBUG] debug_contours is None")

    warped = quad is not None

    # 【关键修复】方向检测移到透视变换之后（在矩形纸张上检测更准确）
    rotation, orient_conf, flip_conf = 0, 0.0, 0.0
    flipped_180 = False
    if options.fix_orientation and min(img.shape[:2]) >= 50:
        rotation, orient_conf, flip_conf = _detect_orientation(img, options.detect_upside_down, options.text_orientation_correction)
        print(f"[DEBUG] 方向检测（透视后）: rotation={rotation}°, confidence={orient_conf:.2f}, flip_confidence={flip_conf:.2f}, text_correction={options.text_orientation_correction}")

        # 增加置信度阈值：只有置信度 > 0.5 才执行旋转
        if rotation and orient_conf > 0.5:
            img = _rotate90(img, rotation)
            steps.append('orientation')
            if rotation == 180:
                flipped_180 = True
                print(f"[DEBUG] 180° 翻转完成，置信度: {flip_conf:.2f}")
            else:
                print(f"[DEBUG] 方向矫正完成: {rotation}°, 置信度: {orient_conf:.2f}")
        elif rotation and orient_conf <= 0.5:
            print(f"[DEBUG] 跳过方向矫正: rotation={rotation}°, confidence={orient_conf:.2f} <= 0.5（置信度过低）")
            rotation = 0  # 重置rotation，表示未旋转
        else:
            print(f"[DEBUG] 无需方向矫正")

    # 生成文字掩码（在原始或透视后的图像上，处理前）
    if min(img.shape[:2]) >= 50:
        try:
            gray_for_mask = _to_gray(img)
            text_mask_img = _text_mask(gray_for_mask, remove_lines=False)
            text_mask_bgr = cv2.cvtColor((text_mask_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            text_mask_encoded = encode_image(text_mask_bgr, 'JPEG', 85)[0]
            debug_images['text_mask_image'] = text_mask_encoded
            print(f"[DEBUG] text_mask_image 编码成功，大小: {len(text_mask_encoded)} bytes, 墨迹密度: {text_mask_img.mean():.4f}")
        except Exception as e:
            print(f"[ERROR] text_mask_image 生成失败: {e}")

    deskew_angle = 0.0
    if options.deskew and min(img.shape[:2]) >= 50:
        print(f"[DEBUG] 去倾斜前图像尺寸: {img.shape[1]}x{img.shape[0]}")
        angle, conf = _estimate_skew(_to_gray(img), 5 if warped else 45)
        print(f"[DEBUG] 倾斜检测: angle={angle:.2f}°, confidence={conf:.3f}, threshold={MIN_SKEW_CONFIDENCE}")
        if abs(angle) >= MIN_DESKEW_ANGLE and conf >= MIN_SKEW_CONFIDENCE:
            # 已透视矫正的纸张只是微调，保持尺寸；否则扩大画布避免裁掉内容
            img = _rotate_image(img, angle, _paper_color(img), expand=not warped)
            deskew_angle = angle
            steps.append('deskew')
            print(f"[DEBUG] 去倾斜完成: {angle:.2f}°, 输出尺寸: {img.shape[1]}x{img.shape[0]}")
        else:
            print(f"[DEBUG] 跳过去倾斜: angle={angle:.2f}° < {MIN_DESKEW_ANGLE}° 或 confidence={conf:.3f} < {MIN_SKEW_CONFIDENCE}，保持文字清晰度")

    if options.trim_borders and not warped:
        before = img.shape[:2]
        img = _crop_to_content(_trim_dark_borders(img))
        if img.shape[:2] != before:
            steps.append('trim')
            print(f"[DEBUG] 裁边完成: {before} -> {img.shape[:2]}")

    # 二值化处理（如果启用）
    if options.binarize:
        img = _apply_binarization(img)
        steps.append('binarize')
        print(f"[DEBUG] 二值化完成")

    if options.manual_rotation:
        img = _rotate90(img, options.manual_rotation)
        steps.append('manual_rotation')
        print(f"[DEBUG] 手动旋转: {options.manual_rotation}°")

    data, mime = encode_image(img, options.output_format, options.quality)

    print(f"[DEBUG] 最终输出: 尺寸={img.shape[1]}x{img.shape[0]}, 格式={options.output_format}, 质量={options.quality}")
    print(f"[DEBUG] 处理完成: steps={steps}, deskew={deskew_angle:.2f}°, orient_conf={orient_conf:.2f}")
    print(f"[DEBUG] debug_images 包含的键: {list(debug_images.keys())}")

    return ProcessResult(
        image_bytes=data,
        mime_type=mime,
        detected_quad=quad.round().astype(int).tolist() if warped else None,
        rotation_applied=(rotation + options.manual_rotation) % 360,
        deskew_angle=deskew_angle,
        orientation_confidence=orient_conf,
        original_size=(w0, h0),
        processed_size=(img.shape[1], img.shape[0]),
        processing_time_ms=(time.time() - t0) * 1000.0,
        steps=steps,
        debug_image_bytes=_debug_image(original, quad) if options.debug else None,
        debug_images=debug_images,
        flipped_180=flipped_180,
        flip_confidence=flip_conf,
        irregular_boundary=irregular_boundary,
        corner_repaired=corner_repaired,
    )


if __name__ == '__main__':
    import json
    import sys

    if len(sys.argv) < 3:
        print('用法: python image_processor.py 输入图片 输出图片(.jpg/.png)')
        sys.exit(1)
    with open(sys.argv[1], 'rb') as f:
        raw = f.read()
    out_fmt = 'PNG' if sys.argv[2].lower().endswith('.png') else 'JPEG'
    result = process_image(raw, ProcessOptions(output_format=out_fmt))
    with open(sys.argv[2], 'wb') as f:
        f.write(result.image_bytes)
    print(json.dumps(result.to_metadata(), ensure_ascii=False, indent=2))
