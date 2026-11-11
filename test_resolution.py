# -*- coding: utf-8 -*-
"""测试分辨率保持效果"""
import sys
from pathlib import Path
from image_processor import process_image, ProcessOptions

def test_resolution_preservation(image_path):
    """测试透视变换是否保持原始分辨率"""
    print(f"\n{'='*60}")
    print(f"测试图像: {image_path}")
    print(f"{'='*60}")

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    options = ProcessOptions(
        detect_document=True,
        deskew=True,
        fix_orientation=True,
        detect_upside_down=False,  # 暂时关闭倒置检测
        trim_borders=True,
        output_format='JPEG',
        quality=95,
        debug=False
    )

    result = process_image(image_bytes, options)

    print(f"\n📊 分辨率对比:")
    print(f"  原始尺寸: {result.original_size[0]}x{result.original_size[1]}")
    print(f"  输出尺寸: {result.processed_size[0]}x{result.processed_size[1]}")

    # 计算分辨率保持率
    original_pixels = result.original_size[0] * result.original_size[1]
    processed_pixels = result.processed_size[0] * result.processed_size[1]
    retention_rate = (processed_pixels / original_pixels) * 100

    print(f"  分辨率保持率: {retention_rate:.1f}%")

    if retention_rate < 70:
        print(f"  ⚠️  警告：分辨率损失过大 ({100-retention_rate:.1f}%)")
    elif retention_rate >= 90:
        print(f"  ✅ 分辨率保持良好")
    else:
        print(f"  ⚠️  分辨率有轻微损失 ({100-retention_rate:.1f}%)")

    print(f"\n🔧 处理步骤: {' → '.join(result.steps)}")
    print(f"⏱️  处理时间: {result.processing_time_ms:.1f}ms")

    if result.detected_quad:
        print(f"📐 检测到纸张四边形: ✅")
    else:
        print(f"📐 检测到纸张四边形: ❌")

    if result.rotation_applied:
        print(f"🔄 旋转角度: {result.rotation_applied}°")

    if result.deskew_angle:
        print(f"📏 去倾斜角度: {result.deskew_angle:.2f}°")

    # 保存输出图像
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_processed.jpg"
    with open(output_path, 'wb') as f:
        f.write(result.image_bytes)
    print(f"\n💾 输出已保存: {output_path}")

    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python test_resolution.py <图像路径>")
        sys.exit(1)

    test_resolution_preservation(sys.argv[1])
