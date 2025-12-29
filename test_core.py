#!/usr/bin/env python3
"""
杂草识别与分析系统核心功能测试脚本
用于测试系统的核心算法，无需GUI界面
"""

import cv2
import numpy as np
import os
from weed_detection import WeedDetectionSystem

def test_core_functionality():
    """测试系统核心功能"""
    print("=== 杂草识别与分析系统核心功能测试 ===")
    
    # 创建系统实例
    weed_system = WeedDetectionSystem()
    
    # 生成测试图像（绿色方块代表植物，棕色背景代表土壤）
    test_image = generate_test_image()
    
    print("1. 测试颜色空间变换...")
    hsv_image = weed_system.color_space_conversion(test_image, 'HSV')
    lab_image = weed_system.color_space_conversion(test_image, 'LAB')
    print("   ✓ 颜色空间变换测试通过")
    
    print("2. 测试植被指数计算...")
    exg_image, cive_image = weed_system.calculate_vegetation_indices(test_image)
    print("   ✓ 植被指数计算测试通过")
    
    print("3. 测试CLAHE增强...")
    clahe_image = weed_system.apply_clahe(test_image)
    print("   ✓ CLAHE增强测试通过")
    
    print("4. 测试大津法阈值分割...")
    binary_image = weed_system.otsu_thresholding(exg_image)
    print("   ✓ 大津法阈值分割测试通过")
    
    print("5. 测试形态学去噪...")
    morph_image = weed_system.morphological_operations(binary_image)
    print("   ✓ 形态学去噪测试通过")
    
    print("6. 测试分水岭算法分割...")
    watershed_image, markers = weed_system.watershed_segmentation(test_image)
    print("   ✓ 分水岭算法分割测试通过")
    
    print("7. 测试轮廓提取与过滤...")
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = weed_system.filter_contours_by_area(contours)
    print(f"   ✓ 轮廓提取与过滤测试通过，找到 {len(filtered_contours)} 个目标")
    
    print("8. 测试形状描述符计算...")
    for i, contour in enumerate(filtered_contours):
        descriptors = weed_system.calculate_shape_descriptors(contour)
        print(f"   目标 {i+1}: 圆度={descriptors['circularity']:.4f}, 长宽比={descriptors['aspect_ratio']:.4f}")
    print("   ✓ 形状描述符计算测试通过")
    
    print("9. 测试质心计算...")
    centroids = weed_system.calculate_centroids(filtered_contours)
    print(f"   ✓ 质心计算测试通过，找到 {len(centroids)} 个质心")
    
    print("10. 测试杂草覆盖度统计...")
    coverage = weed_system.calculate_weed_coverage(morph_image)
    print(f"   ✓ 杂草覆盖度统计测试通过，覆盖度: {coverage:.2f}%")
    
    print("11. 测试色彩矩分析...")
    color_moments = weed_system.calculate_color_moments(test_image, morph_image)
    print("   ✓ 色彩矩分析测试通过")
    
    print("12. 测试纹理特征提取...")
    glcm_features = weed_system.calculate_glcm_features(test_image, morph_image)
    print("   ✓ 纹理特征提取测试通过")
    
    print("13. 测试完整处理流程...")
    results = weed_system.process_single_image(test_image)
    print(f"   ✓ 完整处理流程测试通过")
    print(f"   处理结果：")
    print(f"   - 杂草数量: {results['weed_count']}")
    print(f"   - 覆盖度: {results['coverage']:.2f}%")
    
    print("14. 测试结果标注...")
    annotated_image = weed_system.annotate_results(test_image)
    print("   ✓ 结果标注测试通过")
    
    # 保存测试结果图像
    save_test_results(test_image, hsv_image, exg_image, binary_image, morph_image, watershed_image, annotated_image)
    
    print("\n=== 所有核心功能测试通过！===")

def generate_test_image():
    """生成测试图像"""
    # 创建一个500x500的图像，棕色背景代表土壤
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    image[:, :] = [139, 69, 19]  # 棕色背景
    
    # 绘制几个绿色方块代表植物/杂草
    # 阔叶杂草（圆形）
    cv2.circle(image, (150, 150), 50, (0, 255, 0), -1)
    cv2.circle(image, (350, 150), 40, (0, 200, 0), -1)
    
    # 禾本科杂草（狭长）
    cv2.rectangle(image, (100, 300), (120, 450), (0, 220, 0), -1)
    cv2.rectangle(image, (200, 320), (220, 480), (0, 240, 0), -1)
    cv2.rectangle(image, (350, 300), (370, 450), (0, 180, 0), -1)
    
    return image

def save_test_results(original, hsv, exg, binary, morph, watershed, annotated):
    """保存测试结果图像"""
    # 创建输出目录
    output_dir = "test_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存图像
    cv2.imwrite(os.path.join(output_dir, "original.jpg"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "hsv.jpg"), cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    cv2.imwrite(os.path.join(output_dir, "exg.jpg"), exg)
    cv2.imwrite(os.path.join(output_dir, "binary.jpg"), binary)
    cv2.imwrite(os.path.join(output_dir, "morph.jpg"), morph)
    cv2.imwrite(os.path.join(output_dir, "watershed.jpg"), cv2.cvtColor(watershed, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "annotated.jpg"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    
    print(f"\n测试结果图像已保存到 {output_dir} 目录")

if __name__ == "__main__":
    test_core_functionality()