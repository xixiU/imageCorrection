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

import cv2
import numpy as np

# ---------------- 可调参数 ----------------
MAX_PIXELS = 60000000          # 解码后像素上限，防止超大图耗尽内存
DETECT_MAX_SIDE = 1000         # 纸张检测时的工作分辨率
MIN_DOC_AREA_RATIO = 0.2       # 纸张至少占画面的比例
MAX_DOC_AREA_RATIO = 0.97      # 超过该比例视为纸张铺满画面，不做透视
MIN_EDGE_CONTRAST = 25.0       # 纸张边缘内外 LAB 色差下限，过滤表格框等误检
DOC_INSET_RATIO = 0.004        # 透视前四角向内收缩比例，去掉纸边阴影
DESKEW_MAX_SIDE = 800          # 去倾斜计算分辨率
MIN_DESKEW_ANGLE = 0.3         # 小于该角度不旋转
MIN_SKEW_CONFIDENCE = 0.05     # 投影能量峰值相对中位数的提升下限
ORIENT_MAX_SIDE = 1600         # 方向判断计算分辨率
HORIZONTAL_RATIO = 1.15        # 行/列投影特征比值 >= 该值判为横排
VERTICAL_RATIO = 1.3           # 列/行投影特征比值 >= 该值判为竖排（保守，避免误转）
MIN_PUNCT_FOR_90 = 4           # 判断 90°/270° 时标点证据的最少数量
MIN_PUNCT_FOR_180 = 8          # 判断倒置时标点证据的最少数量
FLIP_180_SCORE = -0.5          # 标点得分低于该值才认为倒置


@dataclass
class ProcessOptions:
    detect_document: bool = True     # 检测纸张边界并透视矫正
    deskew: bool = True              # 去除小角度倾斜
    fix_orientation: bool = True     # 修正 90° 旋转
    detect_upside_down: bool = True  # 修正倒置（证据充分时才翻转）
    trim_borders: bool = True        # 未检测到纸张时裁黑边、裁到内容区域
    manual_rotation: int = 0         # 处理后再顺时针旋转 0/90/180/270
    output_format: str = 'JPEG'      # JPEG / PNG
    quality: int = 90                # JPEG 质量 1-100
    debug: bool = False              # 生成标注了检测四边形的调试图


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


def encode_image(img, fmt='JPEG', quality=90):
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
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR,
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


def _quad_angles_ok(q):
    for i in range(4):
        v1 = q[i - 1] - q[i]
        v2 = q[(i + 1) % 4] - q[i]
        cos_v = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        ang = np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0)))
        if ang < 50 or ang > 130:
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


def _find_document_quad(img_small):
    """在缩小图上找纸张四边形，找不到返回 None。"""
    gray = cv2.GaussianBlur(_to_gray(img_small), (5, 5), 0)
    h, w = gray.shape
    area = float(h * w)
    contours = []

    med = float(np.median(gray))
    for lo, hi in ((0.66 * med, 1.33 * med), (30, 100)):
        edges = cv2.Canny(gray, lo, hi)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cs, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cs)

    # 纸张通常比背景亮：Otsu 二值化后取外轮廓
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(5, int(min(h, w) * 0.02)) | 1
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.extend(cs)

    lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2LAB).astype(np.float32)
    best, best_ratio = None, 0.0
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        if cv2.contourArea(hull) < MIN_DOC_AREA_RATIO * area:
            continue
        quad = _contour_to_quad(hull)
        if quad is None:
            continue
        quad = _order_quad(quad)
        ratio = cv2.contourArea(quad) / area
        # 只接受比当前最优更大的候选，减少色差计算次数
        if ratio < MIN_DOC_AREA_RATIO or ratio > MAX_DOC_AREA_RATIO or ratio <= best_ratio:
            continue
        if not cv2.isContourConvex(quad.reshape(-1, 1, 2)) or not _quad_angles_ok(quad):
            continue
        if _edge_contrast(lab, quad) < MIN_EDGE_CONTRAST:
            continue
        best, best_ratio = quad, ratio
    return best


def _inset_quad(quad, ratio):
    c = quad.mean(axis=0)
    return (quad + (c - quad) * ratio).astype(np.float32)


def _warp_quad(img, quad):
    tl, tr, br, bl = quad
    wd = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    ht = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    wd, ht = max(wd, 1), max(ht, 1)
    dst = np.array([[0, 0], [wd - 1, 0], [wd - 1, ht - 1], [0, ht - 1]], np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, m, (wd, ht), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


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


def _detect_orientation(img, detect_upside_down):
    """返回 (需顺时针旋转的角度, 置信度 0~1)。"""
    small, _ = _resize_max(_to_gray(img), ORIENT_MAX_SIDE)
    mask = _text_mask(small)
    ink = float(mask.mean())
    if ink < 0.002 or ink > 0.4:
        return 0, 0.0
    direction, ratio = _text_direction(mask)

    if direction == 'v':
        # 文字行是竖的：先按顺时针转 90° 评估，得分为负说明应逆时针转
        punct, n, align = _upright_score(cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE))
        total = (punct if n >= MIN_PUNCT_FOR_90 else 0.0) + 0.5 * align
        return (90 if total >= 0 else 270), min(1.0, abs(total))

    if direction == 'h':
        if detect_upside_down:
            punct, n, align = _upright_score(mask)
            # 倒置很少见，要求标点和对齐证据一致且充分才翻转
            if n >= MIN_PUNCT_FOR_180 and punct <= FLIP_180_SCORE and align <= 0.1:
                return 180, min(1.0, -punct)
        return 0, min(1.0, ratio - 1.0)

    return 0, 0.0


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
    """裁到内容区域并保留 2% 页边距；内容过小时不裁。"""
    h, w = img.shape[:2]
    small, scale = _resize_max(_to_gray(img), 1000)
    mask = _text_mask(small, remove_lines=False)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    sh, sw = mask.shape
    ys = np.nonzero(mask.sum(axis=1) > max(2, 0.003 * sw))[0]
    xs = np.nonzero(mask.sum(axis=0) > max(2, 0.003 * sh))[0]
    if len(ys) == 0 or len(xs) == 0:
        return img
    margin = int(max(10, 0.02 * max(h, w)))
    y0 = max(0, int(ys[0] / scale) - margin)
    y1 = min(h, int((ys[-1] + 1) / scale) + margin)
    x0 = max(0, int(xs[0] / scale) - margin)
    x1 = min(w, int((xs[-1] + 1) / scale) + margin)
    if (y1 - y0) * (x1 - x0) < 0.1 * h * w:
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

    if options.detect_document and min(h0, w0) >= 200:
        small, scale = _resize_max(img, DETECT_MAX_SIDE)
        q = _find_document_quad(small)
        if q is not None:
            quad = _inset_quad(q / scale, DOC_INSET_RATIO)
            img = _warp_quad(img, quad)
            steps.append('document_warp')
    warped = quad is not None

    deskew_angle = 0.0
    if options.deskew and min(img.shape[:2]) >= 50:
        angle, conf = _estimate_skew(_to_gray(img), 5 if warped else 45)
        if abs(angle) >= MIN_DESKEW_ANGLE and conf >= MIN_SKEW_CONFIDENCE:
            # 已透视矫正的纸张只是微调，保持尺寸；否则扩大画布避免裁掉内容
            img = _rotate_image(img, angle, _paper_color(img), expand=not warped)
            deskew_angle = angle
            steps.append('deskew')

    rotation, orient_conf = 0, 0.0
    if options.fix_orientation and min(img.shape[:2]) >= 50:
        rotation, orient_conf = _detect_orientation(img, options.detect_upside_down)
        if rotation:
            img = _rotate90(img, rotation)
            steps.append('orientation')

    if options.trim_borders and not warped:
        before = img.shape[:2]
        img = _crop_to_content(_trim_dark_borders(img))
        if img.shape[:2] != before:
            steps.append('trim')

    if options.manual_rotation:
        img = _rotate90(img, options.manual_rotation)
        steps.append('manual_rotation')

    data, mime = encode_image(img, options.output_format, options.quality)
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
