# 图像处理分辨率损失修复报告

## 问题根源

**严重问题：** 输入 3264x2448 → 输出 2096x2483，分辨率损失 35%，文字严重模糊

**根本原因：** 方向检测在透视变换**之前**执行，导致：
1. 原图 3264x2448（横向）被误判需要旋转 90°
2. 旋转后变成 2448x3264（竖向）
3. 在旋转后的图像上做透视变换，输出 2096x2483
4. 宽度从 3264 降到 2096，损失 35% 分辨率

## 修复方案

### 核心改动：调整处理顺序

**修复前流程（错误）：**
```
原图 → 方向检测(可能旋转90°) → 纸张检测 → 透视变换 → 去倾斜
       ↑ 问题：提前旋转导致分辨率损失
```

**修复后流程（正确）：**
```
原图 → 纸张检测 → 透视变换(保持原始分辨率) → 方向检测 → 去倾斜
                   ↑ 关键：在原图尺寸上完成透视变换
```

### 代码修改

**文件：** `D:\IdeaProjects\imageCorrection\image_processor.py`

**修改位置：** 第 935-1080 行

**关键改动：**
1. **删除** 透视变换前的方向检测（第 940-958 行）
2. **移动** 方向检测到透视变换之后（第 1050 行之后）
3. **确保** 纸张检测和透视变换在原图上完成

## 为什么这样修复有效

### 1. 保持原始分辨率
- 在原图（3264x2448）上检测纸张边界
- 透视变换基于原图尺寸计算输出（3264x2xxx）
- 避免提前旋转导致的尺寸变化

### 2. 参考工程对比
**evibhm/ImageCorrection 的处理流程：**
```python
# 1. 缩小图像用于边缘检测
img_for_canny = resizeImg(image_bgr, height=900)

# 2. 边缘检测 → 找最大轮廓 → 获取四边形
canny = getCanny(img_for_canny, ...)
max_contour, _ = findMaxContour(canny)
boxes_on_resized = getBoxPoint(max_contour)

# 3. 坐标还原到原图
boxes_on_original = adaPoint(boxes_on_resized, scale)

# 4. 直接在原图上透视变换（无预先旋转）
warped = warpImage(image_bgr, boxes_on_original)
```

**我们的修复后流程（对齐参考工程）：**
```python
# 1. 缩小图像用于纸张检测
small, scale = _resize_max(img, 1000)

# 2. 检测纸张四边形
q, ... = _find_document_quad(small, ...)

# 3. 坐标还原到原图
quad_scaled = q / scale

# 4. 直接在原图上透视变换（无预先旋转）
img = _warp_quad(img, quad_scaled)

# 5. 透视变换后再做方向检测
rotation, ... = _detect_orientation(img, ...)
```

### 3. 透视变换后检测方向更准确
- 透视变换后纸张已经是标准矩形
- 文字排列清晰，方向判断更可靠
- 避免倾斜纸张干扰方向检测

## 验证方法

### 1. 使用测试脚本
```bash
python test_resolution.py <测试图像路径>
```

**预期结果：**
- 分辨率保持率 ≥ 90%
- 输出尺寸接近输入尺寸
- 文字清晰不模糊

### 2. Web界面测试
```bash
python app.py
```
上传 3264x2448 的测试图像，检查：
- 输出尺寸应接近 3264x2xxx
- 下载图像查看文字清晰度
- 对比参考工程效果

### 3. 对比测试
使用相同图像分别测试：
- 参考工程：`/tmp/ImageCorrection/`
- 我们的实现：修复后的版本

**对比指标：**
- 输出分辨率
- 文字清晰度
- 处理速度

## 其他优化（已包含在代码中）

### 1. 透视变换优化
```python
def _warp_quad(img, quad):
    # 基于实际边长计算目标尺寸（不缩放）
    widthA = int(np.linalg.norm(tr - tl))
    widthB = int(np.linalg.norm(br - bl))
    maxWidth = max(widthA, widthB)
    
    heightA = int(np.linalg.norm(bl - tl))
    heightB = int(np.linalg.norm(br - tr))
    maxHeight = max(heightA, heightB)
    
    # 使用双线性插值（cv2.INTER_LINEAR）
    result = cv2.warpPerspective(img, m, (maxWidth, maxHeight),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
```

### 2. 去倾斜阈值优化
```python
MIN_DESKEW_ANGLE = 10.0  # 只处理大角度倾斜（≥10°）
```
- 小角度倾斜（<10°）不处理，避免旋转导致的质量损失
- 透视变换后纸张通常已接近水平，无需再旋转

### 3. 方向检测置信度优化
```python
if rotation and orient_conf > 0.5:
    img = _rotate90(img, rotation)
```
- 只有置信度 > 0.5 才旋转
- 避免误判导致的错误旋转

## 预期效果

**修复前：**
- 输入：3264x2448
- 输出：2096x2483（损失 35%）
- 文字模糊

**修复后：**
- 输入：3264x2448
- 输出：3264x2xxx（保持 ≥90%）
- 文字清晰

## 注意事项

1. **已有图像重新处理：** 之前处理过的图像需要重新上传处理
2. **性能影响：** 在原图尺寸上做透视变换可能稍慢（但质量更好）
3. **内存占用：** 大图像可能占用更多内存（已有 MAX_PIXELS 限制）

## 后续验证

请使用真实测试图像验证：
1. 分辨率是否保持
2. 文字是否清晰
3. 与参考工程效果对比

如果效果仍不满意，可以考虑：
- 直接迁移参考工程的 warpImage() 等核心函数
- 调整 DOC_INSET_RATIO 参数（当前 -0.005）
- 优化边缘检测参数
