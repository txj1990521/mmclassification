#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试YOLO切图后的长条图搜索问题
直接运行：python test_yolo_cropped_stripe.py
"""

import os
import sys
import cv2
import numpy as np
import json
import time
from pathlib import Path


# ================================
# 直接复制必要的函数（从原搜索脚本）
# ================================

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def imread_unicode(p):
    """读取包含中文路径的图片"""
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def pad_to_square(img_rgb: np.ndarray):
    h, w = img_rgb.shape[:2]
    if h == w:
        return img_rgb
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img_rgb, top, bottom, left, right, cv2.BORDER_REFLECT101)


# ================================
# 配置（从原脚本复制）
# ================================
CONFIG = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_build_zhanlan.py"
CKPT = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"
INDEX_DIR = r"D:\zhanlan\faiss_database_hybrid_new_data"
GLOBAL_META = os.path.join(INDEX_DIR, "global_img_paths.npy")
PATCH_META = os.path.join(INDEX_DIR, "patch_meta.npy")
QUERY_IMG = r"D:\zhanlan\qurrey_data\4.5 TL05027.jpg"
OUT_DIR = r"D:\zhanlan\search_vis\yolo_cropped_test"

# YOLO相关配置
YOLO_SEG_WEIGHTS = r"D:\zhanlanProject\ultralyticsV8\runs\huaxing\exp12\weights\best.pt"
YOLO_DEVICE = 0
SEG_SCORE_THR = 0.6
SEG_USE_CLASSES = None

# ================================
# 导入必要的模块
# ================================
try:
    # 尝试导入必要的YOLO相关函数
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # 这里需要从原搜索脚本导入YOLO切图函数
    # 由于导入可能失败，我们直接复制关键代码
    print("⚠️ 注意：需要从原搜索脚本导入crop_by_mmdet_mask_final函数")
    print("如果导入失败，请手动复制该函数到本脚本")

except ImportError as e:
    print(f"导入警告: {e}")
    print("将继续使用简化版的测试")


# ================================
# 简化版YOLO切图函数（用于测试）
# ================================
def simple_yolo_crop_for_test(img_bgr, debug_dir=None):
    """
    简化版YOLO切图，仅用于测试
    返回: (cropped_img, mask, raw_crop)
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr, None, img_bgr

    H, W = img_bgr.shape[:2]

    # 为了测试，我们模拟YOLO切图的效果
    # 假设切掉周围背景，保留中心区域
    # 模拟长条图：高度远大于宽度

    # 模拟切割参数
    crop_top = int(H * 0.1)  # 切掉顶部10%
    crop_bottom = int(H * 0.1)  # 切掉底部10%
    crop_left = int(W * 0.35)  # 切掉左侧25%
    crop_right = int(W * 0.35)  # 切掉右侧25%

    y1 = crop_top
    y2 = H - crop_bottom
    x1 = crop_left
    x2 = W - crop_right

    # 确保裁剪区域有效
    if y2 <= y1 or x2 <= x1:
        return img_bgr, None, img_bgr

    # 裁剪
    crop_img = img_bgr[y1:y2, x1:x2].copy()

    # 创建模拟mask（全白，表示整个裁剪区域都是前景）
    mask = np.ones((crop_img.shape[0], crop_img.shape[1]), dtype=np.uint8) * 255

    # 保存调试信息
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "test_cropped.jpg"), crop_img)
        cv2.imwrite(os.path.join(debug_dir, "test_mask.jpg"), mask)

    print(f"模拟YOLO切图: 原始尺寸 {W}x{H} -> 裁剪后 {crop_img.shape[1]}x{crop_img.shape[0]}")
    print(f"宽高比变化: {H / W:.2f} -> {crop_img.shape[0] / crop_img.shape[1]:.2f}")

    return crop_img, mask, crop_img


# ================================
# 主测试函数
# ================================

def test_yolo_cropping_effect():
    """测试YOLO切图的效果"""
    print("=" * 80)
    print("测试YOLO切图对搜索的影响")
    print("=" * 80)

    # 1. 创建输出目录
    ensure_dir(OUT_DIR)

    # 2. 加载原始query图片
    print("\n📁 加载原始query图片...")
    original_img = imread_unicode(QUERY_IMG)
    if original_img is None:
        print("❌ 无法加载query图片")
        return

    original_h, original_w = original_img.shape[:2]
    original_aspect = original_h / original_w
    print(f"原始图片: {original_w}x{original_h} (宽高比: {original_aspect:.2f})")

    # 保存原始图片
    cv2.imwrite(os.path.join(OUT_DIR, "00_original.jpg"), original_img)

    # 3. 模拟YOLO切图
    print("\n✂️ 模拟YOLO切图...")
    cropped_img, mask, raw_crop = simple_yolo_crop_for_test(original_img, OUT_DIR)

    if cropped_img is None:
        print("❌ 切图失败")
        return

    cropped_h, cropped_w = cropped_img.shape[:2]
    cropped_aspect = cropped_h / cropped_w

    print(f"切图后: {cropped_w}x{cropped_h} (宽高比: {cropped_aspect:.2f})")
    print(f"宽高比变化: {original_aspect:.2f} → {cropped_aspect:.2f}")

    # 判断是否为长条图
    is_vertical_stripe = cropped_aspect >= 3.0
    print(f"是否为竖向长条图: {is_vertical_stripe} ({'是' if is_vertical_stripe else '否'})")

    # 4. 分析切图对patch提取的影响
    print("\n📐 分析patch提取影响...")

    # 模拟不同patch提取策略
    patch_size = 224
    stride_normal = int(patch_size * 0.5)  # 正常步长
    stride_dense = int(patch_size * 0.25)  # 密集步长

    # 正常策略（网格提取）
    if cropped_w >= patch_size and cropped_h >= patch_size:
        patches_normal_h = max(1, (cropped_h - patch_size) // stride_normal + 1)
        patches_normal_w = max(1, (cropped_w - patch_size) // stride_normal + 1)
        patches_normal_total = patches_normal_h * patches_normal_w
    else:
        patches_normal_total = 1  # 图片太小，只能取1个patch

    # 长条图专用策略（沿高度密集采样）
    if is_vertical_stripe and cropped_h >= patch_size:
        # 沿高度密集采样
        patches_vertical_h = max(1, (cropped_h - patch_size) // stride_dense + 1)

        # 宽度方向采样多个位置
        if cropped_w >= patch_size * 3:
            # 宽度足够：左、中、右
            patches_vertical_w = 3
        elif cropped_w >= patch_size:
            # 宽度足够但不多：中心
            patches_vertical_w = 1
        else:
            # 宽度不足：可能无有效patch
            patches_vertical_w = 0

        patches_vertical_total = patches_vertical_h * patches_vertical_w
    else:
        patches_vertical_total = patches_normal_total

    print(f"正常patch提取策略: 约{patches_normal_total}个patches")
    print(f"长条图专用策略: 约{patches_vertical_total}个patches")

    # 5. 可视化patch提取位置
    print("\n🖼️ 可视化patch提取位置...")
    visualize_patch_strategies(cropped_img, OUT_DIR)

    # 6. 检查数据库中的目标图片
    print("\n🔍 检查数据库...")
    try:
        img_paths = np.load(GLOBAL_META, allow_pickle=True)
        print(f"数据库图片数量: {len(img_paths)}")

        # 查找目标图片
        target_name = "TL05027"
        target_indices = []

        for i, p in enumerate(img_paths):
            if target_name in os.path.basename(str(p)):
                target_indices.append(i)

        if not target_indices:
            print(f"❌ 未找到目标图片: {target_name}")
            return

        target_idx = target_indices[0]
        target_path = img_paths[target_idx]
        print(f"✅ 找到目标图片: {os.path.basename(str(target_path))}")

        # 加载目标图片
        target_img = imread_unicode(target_path)
        if target_img is None:
            print("❌ 无法加载目标图片")
            return

        target_h, target_w = target_img.shape[:2]
        target_aspect = target_h / target_w

        print(f"目标图片: {target_w}x{target_h} (宽高比: {target_aspect:.2f})")

        # 保存目标图片
        cv2.imwrite(os.path.join(OUT_DIR, "target_original.jpg"), target_img)

        # 比较query切图后与目标的尺寸
        print(f"\n📏 尺寸比较:")
        print(f"Query切图后: {cropped_w}x{cropped_h} (宽高比: {cropped_aspect:.2f})")
        print(f"目标图片: {target_w}x{target_h} (宽高比: {target_aspect:.2f})")

        # 检查宽高比差异
        aspect_diff = abs(cropped_aspect - target_aspect)
        print(f"宽高比差异: {aspect_diff:.2f}")

        if aspect_diff > 2.0:
            print("⚠️  宽高比差异较大，可能影响匹配")

    except Exception as e:
        print(f"❌ 加载数据库失败: {e}")

    # 7. 分析问题原因
    print("\n🔎 问题分析...")

    issues = []

    # 检查1: 是否为长条图
    if is_vertical_stripe:
        issues.append("Query切图后成为竖向长条图")

    # 检查2: patch提取数量
    if patches_normal_total < 10:
        issues.append(f"正常patch提取策略可能patch数量不足 ({patches_normal_total}个)")

    # 检查3: 尺寸匹配
    if 'target_aspect' in locals() and aspect_diff > 2.0:
        issues.append(f"Query与目标宽高比差异较大 ({aspect_diff:.2f})")

    if issues:
        print("发现以下可能问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("未发现明显问题")

    # 8. 生成修复建议
    print("\n💡 修复建议:")

    if is_vertical_stripe:
        print("1. 使用专门的竖向长条图patch提取策略")
        print("2. 修改get_query_patch_feats函数，增加对竖向长条图的处理")
        print("3. 沿高度方向密集采样（stride设为patch_size的0.25-0.3倍）")
        print("4. 在宽度方向采样多个位置（如果宽度允许）")
        print("5. 增加最大patch数量到200-300个")

    if patches_normal_total < 10:
        print("6. 降低patch尺寸或使用重叠度更高的采样")
        print("7. 考虑使用多尺度patch提取")

    # 9. 生成对比图
    print("\n📊 生成对比图...")
    generate_comparison_chart(original_img, cropped_img, OUT_DIR)

    # 10. 保存测试结果
    save_test_results({
        "original_size": [original_w, original_h],
        "original_aspect": float(original_aspect),
        "cropped_size": [cropped_w, cropped_h],
        "cropped_aspect": float(cropped_aspect),
        "is_vertical_stripe": is_vertical_stripe,
        "patches_normal": int(patches_normal_total),
        "patches_vertical": int(patches_vertical_total),
        "issues": issues,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }, OUT_DIR)

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
    print(f"详细结果请查看: {OUT_DIR}")


def visualize_patch_strategies(img, out_dir):
    """可视化不同patch提取策略"""
    h, w = img.shape[:2]
    patch_size = 224

    # 创建可视化图像
    vis = img.copy()

    # 策略1: 正常网格提取
    vis_normal = vis.copy()
    stride_normal = patch_size // 2

    if w >= patch_size and h >= patch_size:
        y = 0
        patch_count = 0
        while y + patch_size <= h and patch_count < 5:  # 只显示前5行
            x = 0
            while x + patch_size <= w and patch_count < 20:  # 总共最多20个
                cv2.rectangle(vis_normal, (x, y), (x + patch_size, y + patch_size), (0, 255, 0), 2)
                x += stride_normal
                patch_count += 1
            y += stride_normal

    cv2.putText(vis_normal, "Normal Grid Strategy", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "patch_strategy_normal.jpg"), vis_normal)

    # 策略2: 竖向长条图专用策略
    if h / w >= 3.0:  # 如果是长条图
        vis_vertical = vis.copy()

        # 沿高度密集采样
        stride_dense = patch_size // 4

        # 宽度方向采样位置
        width_positions = []
        if w >= patch_size * 3:
            width_positions = [0, w // 2 - patch_size // 2, w - patch_size]
        elif w >= patch_size:
            width_positions = [max(0, w // 2 - patch_size // 2)]

        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0)]

        for i, x in enumerate(width_positions):
            if x < 0 or x > w - patch_size:
                continue

            color = colors[i % len(colors)]
            y = 0
            patch_count = 0

            while y + patch_size <= h and patch_count < 10:
                cv2.rectangle(vis_vertical, (x, y), (x + patch_size, y + patch_size), color, 2)
                y += stride_dense
                patch_count += 1

        cv2.putText(vis_vertical, "Vertical Stripe Strategy", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, "patch_strategy_vertical.jpg"), vis_vertical)

    print(f"📸 Patch策略可视化已保存到 {out_dir}")


def generate_comparison_chart(original_img, cropped_img, out_dir):
    """生成对比图表"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 原始图片
    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Original: {original_img.shape[1]}x{original_img.shape[0]}')
    axes[0, 0].axis('off')

    # 2. 切图后图片
    axes[0, 1].imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'After YOLO Crop: {cropped_img.shape[1]}x{cropped_img.shape[0]}')
    axes[0, 1].axis('off')

    # 3. 尺寸对比条形图
    sizes = [
        ['Width', original_img.shape[1], cropped_img.shape[1]],
        ['Height', original_img.shape[0], cropped_img.shape[0]]
    ]

    x = np.arange(2)
    width = 0.35

    axes[1, 0].bar(x - width / 2, [original_img.shape[1], original_img.shape[0]],
                   width, label='Original', color='skyblue')
    axes[1, 0].bar(x + width / 2, [cropped_img.shape[1], cropped_img.shape[0]],
                   width, label='Cropped', color='lightcoral')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Width', 'Height'])
    axes[1, 0].set_ylabel('Pixels')
    axes[1, 0].set_title('Size Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 宽高比对比
    original_aspect = original_img.shape[0] / original_img.shape[1]
    cropped_aspect = cropped_img.shape[0] / cropped_img.shape[1]

    aspects = [original_aspect, cropped_aspect]
    labels = ['Original', 'Cropped']
    colors = ['skyblue', 'lightcoral']

    bars = axes[1, 1].bar(labels, aspects, color=colors)
    axes[1, 1].set_ylabel('Aspect Ratio (H/W)')
    axes[1, 1].set_title('Aspect Ratio Comparison')
    axes[1, 1].grid(True, alpha=0.3)

    # 在柱状图上添加数值
    for bar, aspect in zip(bars, aspects):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{aspect:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📈 对比图表已保存: {os.path.join(out_dir, 'comparison_chart.png')}")


def save_test_results(results, out_dir):
    """保存测试结果"""
    # 保存为JSON
    json_file = os.path.join(out_dir, 'test_results.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存为文本报告
    report_file = os.path.join(out_dir, 'test_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLO切图长条图搜索问题测试报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 测试概况\n")
        f.write("-" * 40 + "\n")
        f.write(f"测试时间: {results['timestamp']}\n")
        f.write(f"Query图片: {QUERY_IMG}\n\n")

        f.write("2. 图片尺寸分析\n")
        f.write("-" * 40 + "\n")
        f.write(f"原始尺寸: {results['original_size'][0]}x{results['original_size'][1]}\n")
        f.write(f"原始宽高比: {results['original_aspect']:.2f}\n")
        f.write(f"切图后尺寸: {results['cropped_size'][0]}x{results['cropped_size'][1]}\n")
        f.write(f"切图后宽高比: {results['cropped_aspect']:.2f}\n")
        f.write(f"是否为竖向长条图: {results['is_vertical_stripe']}\n\n")

        f.write("3. Patch提取分析\n")
        f.write("-" * 40 + "\n")
        f.write(f"正常策略预估patch数: {results['patches_normal']}\n")
        f.write(f"长条图策略预估patch数: {results['patches_vertical']}\n\n")

        f.write("4. 发现的问题\n")
        f.write("-" * 40 + "\n")
        if results['issues']:
            for i, issue in enumerate(results['issues'], 1):
                f.write(f"{i}. {issue}\n")
        else:
            f.write("未发现明显问题\n")

        f.write("\n5. 修复建议\n")
        f.write("-" * 40 + "\n")
        if results['is_vertical_stripe']:
            f.write("1. 修改主搜索脚本中的patch提取策略\n")
            f.write("2. 在get_query_patch_feats函数中添加对竖向长条图的特殊处理\n")
            f.write("3. 建议修改参数:\n")
            f.write("   - stride_ratio: 0.25 (更密集)\n")
            f.write("   - max_patches: 256 (更多patch)\n")
            f.write("   - min_mask_cover: 0.05 (更低门槛)\n")
            f.write("4. 考虑添加专门的竖向长条图patch提取函数\n")

    print(f"📄 测试报告已保存: {report_file}")


# ================================
# 主函数
# ================================
def main():
    """主函数"""
    print("YOLO切图长条图搜索问题诊断")
    print(f"开始时间: {time.strftime('%H:%M:%S')}")

    try:
        test_yolo_cropping_effect()
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# ================================
# 脚本入口
# ================================
if __name__ == "__main__":
    # 直接运行，不要用pytest
    main()