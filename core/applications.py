"""
应用模块
包含应用与数据管理功能
"""

import cv2
import numpy as np
import os
import glob
import pandas as pd
from .utils import validate_image, ensure_rgb
from .config import Config

class Applications:
    """
    应用与数据管理模块
    """
    
    def __init__(self, config=None):
        """
        初始化应用模块
        
        Args:
            config: 配置对象，包含批量处理等参数
        """
        self.config = config or Config()
    
    def calibrate_ruler(self, image, known_distance, pixel_distance):
        """
        标尺校准与实际面积转换
        
        Args:
            image: 输入图像（用于显示，不影响校准计算）
            known_distance: 已知实际距离（厘米）
            pixel_distance: 图像中对应的像素距离
        
        Returns:
            float: 像素到厘米的转换比例（厘米/像素）
        """
        # 验证图像
        valid, error_msg = validate_image(image)
        if not valid:
            raise ValueError(f"无效图像: {error_msg}")
        
        # 验证输入参数
        if known_distance <= 0:
            raise ValueError("已知距离必须大于0")
        
        if pixel_distance <= 0:
            raise ValueError("像素距离必须大于0")
        
        # 计算像素到厘米的转换比例
        pixel_to_cm = known_distance / pixel_distance
        
        return pixel_to_cm
    
    def pixel_to_actual_area(self, pixel_area, pixel_to_cm_ratio):
        """
        将像素面积转换为实际面积（平方厘米）
        
        Args:
            pixel_area: 像素面积
            pixel_to_cm_ratio: 像素到厘米的转换比例（厘米/像素）
        
        Returns:
            float: 实际面积（平方厘米）
        """
        if pixel_area < 0:
            raise ValueError("像素面积不能为负数")
        
        if pixel_to_cm_ratio <= 0:
            raise ValueError("转换比例必须大于0")
        
        # 计算实际面积（平方厘米）
        actual_area = pixel_area * (pixel_to_cm_ratio ** 2)
        
        return actual_area
    
    def batch_process(self, input_folder, output_folder, detector):
        """
        批量处理图像
        
        Args:
            input_folder: 输入图像文件夹路径
            output_folder: 输出结果文件夹路径
            detector: 杂草检测系统实例
        
        Returns:
            list: 处理结果列表
        """
        # 验证输入文件夹
        if not os.path.exists(input_folder):
            raise ValueError(f"输入文件夹不存在: {input_folder}")
        
        # 创建输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 获取所有图像文件
        image_files = glob.glob(os.path.join(input_folder, '*.jpg')) + \
                     glob.glob(os.path.join(input_folder, '*.jpeg')) + \
                     glob.glob(os.path.join(input_folder, '*.png')) + \
                     glob.glob(os.path.join(input_folder, '*.bmp'))
        
        if not image_files:
            raise ValueError(f"输入文件夹中没有图像文件: {input_folder}")
        
        results_list = []
        
        for image_path in image_files:
            try:
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # 转换为RGB格式
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 处理图像
                results = detector.process_single_image(image_rgb)
                
                # 保存结果
                results['image_name'] = os.path.basename(image_path)
                results_list.append(results)
                
                # 生成标注图像
                annotated_image = detector.annotate_results(image_rgb)
                
                # 保存标注图像
                output_image_name = self.config.batch_output_format.format(os.path.basename(image_path))
                output_image_path = os.path.join(output_folder, output_image_name)
                cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {str(e)}")
                continue
        
        # 导出结果到CSV
        self.export_results_to_csv(results_list, output_folder)
        
        return results_list
    
    def export_results_to_csv(self, results_list, output_folder):
        """
        将结果导出到CSV文件
        
        Args:
            results_list: 处理结果列表
            output_folder: 输出文件夹路径
        """
        if not results_list:
            return
        
        # 准备CSV数据
        csv_data = []
        
        for result in results_list:
            # 提取基本信息
            image_name = result.get('image_name', '')
            weed_count = result.get('weed_count', 0)
            coverage = result.get('coverage', 0.0)
            
            # 提取形状描述符统计信息
            shape_stats = result.get('shape_stats', {})
            mean_circularity = shape_stats.get('mean_circularity', 0.0)
            mean_aspect_ratio = shape_stats.get('mean_aspect_ratio', 0.0)
            mean_perimeter_area_ratio = shape_stats.get('mean_perimeter_area_ratio', 0.0)
            
            # 添加到CSV数据
            csv_data.append({
                'image_name': image_name,
                'weed_count': weed_count,
                'coverage': coverage,
                'mean_circularity': mean_circularity,
                'mean_aspect_ratio': mean_aspect_ratio,
                'mean_perimeter_area_ratio': mean_perimeter_area_ratio
            })
        
        # 创建DataFrame并导出
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_folder, 'weed_analysis_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
    
    def annotate_results(self, image, contours, centroids, results):
        """
        结果实时标注
        
        Args:
            image: 输入图像（RGB格式）
            contours: 检测到的杂草轮廓列表
            centroids: 质心坐标列表
            results: 处理结果字典
        
        Returns:
            np.ndarray: 标注后的图像
        """
        # 验证图像
        valid, error_msg = validate_image(image)
        if not valid:
            raise ValueError(f"无效图像: {error_msg}")
        
        # 确保RGB格式
        rgb_image = ensure_rgb(image)
        
        # 创建标注图像副本
        annotated = rgb_image.copy()
        
        # 绘制轮廓
        cv2.drawContours(annotated, contours, -1, (255, 0, 0), 2)
        
        # 绘制质心
        for centroid in centroids:
            cv2.circle(annotated, centroid, 5, (0, 255, 0), -1)
        
        # 添加文字信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        text_color = (255, 255, 255)
        text_bg_color = (0, 0, 0)
        
        # 杂草数量
        weed_count = results.get('weed_count', 0)
        text = f"Weed Count: {weed_count}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(annotated, (10, 10), (10 + text_size[0], 10 + text_size[1] + 10), text_bg_color, -1)
        cv2.putText(annotated, text, (10, 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # 覆盖度
        coverage = results.get('coverage', 0.0)
        text = f"Coverage: {coverage:.2f}%"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(annotated, (10, 40), (10 + text_size[0], 40 + text_size[1] + 10), text_bg_color, -1)
        cv2.putText(annotated, text, (10, 60), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # 平均圆度
        shape_stats = results.get('shape_stats', {})
        mean_circularity = shape_stats.get('mean_circularity', 0.0)
        text = f"Mean Circularity: {mean_circularity:.4f}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(annotated, (10, 70), (10 + text_size[0], 70 + text_size[1] + 10), text_bg_color, -1)
        cv2.putText(annotated, text, (10, 90), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        return annotated
    
    def measure_object_size(self, image, contour, pixel_to_cm_ratio):
        """
        测量目标物体的实际尺寸
        
        Args:
            image: 输入图像（用于显示，不影响测量）
            contour: 目标轮廓
            pixel_to_cm_ratio: 像素到厘米的转换比例
        
        Returns:
            dict: 包含长度、宽度、面积的测量结果
        """
        # 验证图像
        valid, error_msg = validate_image(image)
        if not valid:
            raise ValueError(f"无效图像: {error_msg}")
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 转换为实际尺寸
        actual_width = w * pixel_to_cm_ratio
        actual_height = h * pixel_to_cm_ratio
        
        # 计算实际面积
        pixel_area = cv2.contourArea(contour)
        actual_area = self.pixel_to_actual_area(pixel_area, pixel_to_cm_ratio)
        
        return {
            'width_cm': actual_width,
            'height_cm': actual_height,
            'area_cm2': actual_area
        }