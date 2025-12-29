"""
杂草检测器模块
整合所有功能模块，提供统一的检测接口
"""

import cv2
import numpy as np
from .image_processing import ImageProcessing
from .segmentation import Segmentation
from .feature_analysis import FeatureAnalysis
from .statistical_analysis import StatisticalAnalysis
from .applications import Applications
from .config import Config
from .utils import setup_logger, validate_image, ensure_rgb

class WeedDetector:
    """
    杂草检测系统主类
    整合所有功能模块，提供统一的检测接口
    """
    
    def __init__(self, config=None):
        """
        初始化杂草检测系统
        
        Args:
            config: 配置对象
        """
        # 初始化配置
        self.config = config or Config()
        
        # 设置日志
        self.logger = setup_logger(
            name='WeedDetector',
            log_dir=self.config.log_dir,
            level=self.config.log_level
        )
        
        # 初始化各个功能模块
        self.image_processor = ImageProcessing(self.config)
        self.segmentor = Segmentation(self.config)
        self.feature_analyzer = FeatureAnalysis(self.config)
        self.stat_analyzer = StatisticalAnalysis()
        self.applications = Applications(self.config)
        
        # 存储处理结果
        self.original_image = None
        self.processed_image = None
        self.hsv_image = None
        self.lab_image = None
        self.exg_image = None
        self.cive_image = None
        self.clahe_image = None
        self.binary_image = None
        self.morph_image = None
        self.watershed_image = None
        self.contours = []
        self.filtered_contours = []
        self.centroids = []
        self.results = {}
        
        self.logger.info("杂草检测系统初始化完成")
    
    def process_single_image(self, image):
        """
        处理单张图像
        
        Args:
            image: 输入图像（RGB格式）
        
        Returns:
            dict: 处理结果字典
        """
        try:
            self.logger.info("开始处理图像")
            
            # 验证图像
            valid, error_msg = validate_image(image)
            if not valid:
                raise ValueError(f"无效图像: {error_msg}")
            
            # 确保RGB格式
            self.original_image = ensure_rgb(image)
            
            # 1. 颜色空间变换
            self.hsv_image = self.image_processor.color_space_conversion(self.original_image, 'HSV')
            self.lab_image = self.image_processor.color_space_conversion(self.original_image, 'LAB')
            self.logger.debug("颜色空间变换完成")
            
            # 2. 植被指数计算
            self.exg_image, self.cive_image = self.image_processor.calculate_vegetation_indices(self.original_image)
            self.logger.debug("植被指数计算完成")
            
            # 3. CLAHE增强
            self.clahe_image = self.image_processor.apply_clahe(self.original_image)
            self.logger.debug("CLAHE增强完成")
            
            # 4. 大津法阈值分割
            self.binary_image = self.segmentor.otsu_thresholding(self.exg_image)
            self.logger.debug("大津法阈值分割完成")
            
            # 5. 形态学去噪
            self.morph_image = self.segmentor.morphological_operations(self.binary_image)
            self.logger.debug("形态学去噪完成")
            
            # 6. 分水岭算法分割
            self.watershed_image, _ = self.segmentor.watershed_segmentation(self.original_image)
            self.logger.debug("分水岭算法分割完成")
            
            # 7. 提取轮廓
            self.contours = self.segmentor.extract_contours(self.morph_image)
            self.logger.debug(f"提取到 {len(self.contours)} 个轮廓")
            
            # 8. 轮廓过滤
            self.filtered_contours = self.feature_analyzer.filter_contours_by_area(self.contours)
            self.logger.debug(f"过滤后剩余 {len(self.filtered_contours)} 个轮廓")
            
            # 9. 计算质心
            self.centroids = self.feature_analyzer.calculate_centroids(self.filtered_contours)
            self.logger.debug(f"计算到 {len(self.centroids)} 个质心")
            
            # 10. 计算形状描述符
            shape_descriptors = []
            for contour in self.filtered_contours:
                descriptors = self.feature_analyzer.calculate_shape_descriptors(contour)
                shape_descriptors.append(descriptors)
            self.logger.debug("形状描述符计算完成")
            
            # 11. 计算杂草覆盖度
            coverage = self.stat_analyzer.calculate_weed_coverage(self.morph_image)
            self.logger.debug(f"杂草覆盖度: {coverage:.2f}%")
            
            # 12. 计算色彩矩
            color_moments = self.stat_analyzer.calculate_color_moments(self.original_image, self.morph_image)
            self.logger.debug("色彩矩计算完成")
            
            # 13. 计算纹理特征
            glcm_features = self.stat_analyzer.calculate_glcm_features(self.original_image, self.morph_image)
            self.logger.debug("纹理特征计算完成")
            
            # 14. 计算形状描述符统计
            shape_stats = self.stat_analyzer.calculate_statistical_summary(shape_descriptors)
            self.logger.debug("形状描述符统计完成")
            
            # 保存结果
            self.results = {
                'weed_count': len(self.filtered_contours),
                'coverage': coverage,
                'shape_descriptors': shape_descriptors,
                'shape_stats': shape_stats,
                'color_moments': color_moments,
                'glcm_features': glcm_features,
                'contours': self.filtered_contours,
                'centroids': self.centroids
            }
            
            self.logger.info("图像处理完成")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"图像处理失败: {str(e)}", exc_info=True)
            raise
    
    def annotate_results(self, image=None):
        """
        标注处理结果
        
        Args:
            image: 可选，输入图像，默认使用原始图像
        
        Returns:
            np.ndarray: 标注后的图像
        """
        try:
            if image is None:
                if self.original_image is None:
                    raise ValueError("没有可用的图像进行标注")
                image = self.original_image
            
            # 确保RGB格式
            rgb_image = ensure_rgb(image)
            
            # 标注结果
            annotated = self.applications.annotate_results(
                rgb_image,
                self.filtered_contours,
                self.centroids,
                self.results
            )
            
            return annotated
            
        except Exception as e:
            self.logger.error(f"结果标注失败: {str(e)}", exc_info=True)
            raise
    
    def batch_process(self, input_folder, output_folder):
        """
        批量处理图像
        
        Args:
            input_folder: 输入图像文件夹路径
            output_folder: 输出结果文件夹路径
        
        Returns:
            list: 处理结果列表
        """
        try:
            self.logger.info(f"开始批量处理，输入文件夹: {input_folder}, 输出文件夹: {output_folder}")
            
            results_list = self.applications.batch_process(input_folder, output_folder, self)
            
            self.logger.info(f"批量处理完成，共处理 {len(results_list)} 张图像")
            
            return results_list
            
        except Exception as e:
            self.logger.error(f"批量处理失败: {str(e)}", exc_info=True)
            raise
    
    def calibrate(self, known_distance, pixel_distance):
        """
        标尺校准
        
        Args:
            known_distance: 已知实际距离（厘米）
            pixel_distance: 图像中对应的像素距离
        
        Returns:
            float: 像素到厘米的转换比例
        """
        try:
            if self.original_image is None:
                raise ValueError("没有可用的图像进行校准")
            
            pixel_to_cm = self.applications.calibrate_ruler(
                self.original_image,
                known_distance,
                pixel_distance
            )
            
            self.logger.info(f"校准完成，转换比例: {pixel_to_cm} 厘米/像素")
            
            return pixel_to_cm
            
        except Exception as e:
            self.logger.error(f"校准失败: {str(e)}", exc_info=True)
            raise
    
    def measure_weed_sizes(self, pixel_to_cm_ratio):
        """
        测量杂草实际尺寸
        
        Args:
            pixel_to_cm_ratio: 像素到厘米的转换比例
        
        Returns:
            list: 杂草尺寸列表
        """
        try:
            if not self.filtered_contours:
                raise ValueError("没有可用的杂草轮廓进行测量")
            
            weed_sizes = []
            for contour in self.filtered_contours:
                size = self.applications.measure_object_size(
                    self.original_image,
                    contour,
                    pixel_to_cm_ratio
                )
                weed_sizes.append(size)
            
            self.logger.info(f"测量完成，共测量 {len(weed_sizes)} 个杂草")
            
            return weed_sizes
            
        except Exception as e:
            self.logger.error(f"尺寸测量失败: {str(e)}", exc_info=True)
            raise
    
    def get_intermediate_results(self):
        """
        获取中间处理结果
        
        Returns:
            dict: 中间结果字典
        """
        return {
            'original_image': self.original_image,
            'hsv_image': self.hsv_image,
            'lab_image': self.lab_image,
            'exg_image': self.exg_image,
            'cive_image': self.cive_image,
            'clahe_image': self.clahe_image,
            'binary_image': self.binary_image,
            'morph_image': self.morph_image,
            'watershed_image': self.watershed_image,
            'contours': self.contours,
            'filtered_contours': self.filtered_contours,
            'centroids': self.centroids
        }
    
    def reset(self):
        """
        重置系统状态
        """
        self.original_image = None
        self.processed_image = None
        self.hsv_image = None
        self.lab_image = None
        self.exg_image = None
        self.cive_image = None
        self.clahe_image = None
        self.binary_image = None
        self.morph_image = None
        self.watershed_image = None
        self.contours = []
        self.filtered_contours = []
        self.centroids = []
        self.results = {}
        
        self.logger.info("系统状态已重置")