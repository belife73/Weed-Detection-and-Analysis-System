#!/usr/bin/env python3
"""
命令行测试脚本
用于在没有GUI的环境中测试核心功能模块
"""

import cv2
import numpy as np
import os
from core import WeedDetector, Config, setup_logger

def test_core_functionality():
    """测试核心功能"""
    # 设置日志
    logger = setup_logger('TestLogger', 'logs')
    logger.info("=== 开始测试杂草识别与分析系统核心功能 ===")
    
    # 生成测试图像
    logger.info("生成测试图像")
    test_image = generate_test_image()
    
    # 创建配置对象
    config = Config()
    
    # 创建杂草检测器
    logger.info("初始化杂草检测器")
    detector = WeedDetector(config)
    
    # 处理图像
    logger.info("开始处理测试图像")
    results = detector.process_single_image(test_image)
    
    # 生成标注图像
    logger.info("生成标注图像")
    annotated_image = detector.annotate_results()
    
    # 保存测试结果
    logger.info("保存测试结果")
    save_test_results(test_image, annotated_image, results)
    
    # 打印结果摘要
    logger.info("=== 测试结果摘要 ===")
    logger.info(f"杂草数量: {results.get('weed_count', 0)}")
    logger.info(f"覆盖度: {results.get('coverage', 0.0):.2f}%")
    
    shape_stats = results.get('shape_stats', {})
    if shape_stats:
        logger.info(f"平均圆度: {shape_stats.get('mean_circularity', 0.0):.4f}")
        logger.info(f"平均长宽比: {shape_stats.get('mean_aspect_ratio', 0.0):.4f}")
    
    logger.info("=== 所有测试完成！===")

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

def save_test_results(original, annotated, results):
    """保存测试结果"""
    # 创建输出目录
    output_dir = "cli_test_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存原始图像
    original_path = os.path.join(output_dir, "original.jpg")
    cv2.imwrite(original_path, cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    
    # 保存标注图像
    annotated_path = os.path.join(output_dir, "annotated.jpg")
    cv2.imwrite(annotated_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    
    # 保存结果文本
    result_path = os.path.join(output_dir, "results.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=== 杂草分析结果 ===\n")
        f.write(f"杂草数量: {results.get('weed_count', 0)}\n")
        f.write(f"覆盖度: {results.get('coverage', 0.0):.2f}%\n")
        
        shape_stats = results.get('shape_stats', {})
        if shape_stats:
            f.write("\n形状描述符统计:\n")
            f.write(f"平均圆度: {shape_stats.get('mean_circularity', 0.0):.4f}\n")
            f.write(f"平均长宽比: {shape_stats.get('mean_aspect_ratio', 0.0):.4f}\n")
            f.write(f"平均周长面积比: {shape_stats.get('mean_perimeter_area_ratio', 0.0):.4f}\n")
        
        f.write("\n详细形状描述符:\n")
        shape_descriptors = results.get('shape_descriptors', [])
        for i, desc in enumerate(shape_descriptors):
            f.write(f"\n杂草 {i+1}:\n")
            f.write(f"  面积: {desc.get('area', 0.0):.2f} 像素\n")
            f.write(f"  周长: {desc.get('perimeter', 0.0):.2f} 像素\n")
            f.write(f"  圆度: {desc.get('circularity', 0.0):.4f}\n")
            f.write(f"  长宽比: {desc.get('aspect_ratio', 0.0):.4f}\n")

if __name__ == "__main__":
    test_core_functionality()