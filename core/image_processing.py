"""
图像处理模块
包含图像增强与色彩转换功能
"""

import cv2
import numpy as np
from .utils import validate_image, ensure_rgb, normalize_image
from .config import Config

class ImageProcessing:
    """
    图像增强与色彩转换模块
    """
    
    def __init__(self, config=None):
        """
        初始化图像处理模块
        
        Args:
            config: 配置对象，包含CLAHE等参数
        """
        self.config = config or Config()
    
    def color_space_conversion(self, image, space='HSV'):
        """
        颜色空间变换
        
        Args:
            image: 输入图像（RGB格式）
            space: 目标颜色空间，可选值：'HSV', 'LAB', 'RGB'
        
        Returns:
            np.ndarray: 转换后的图像
        """
        # 验证图像
        valid, error_msg = validate_image(image)
        if not valid:
            raise ValueError(f"无效图像: {error_msg}")
        
        # 确保RGB格式
        rgb_image = ensure_rgb(image)
        
        # 颜色空间转换
        if space == 'HSV':
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        elif space == 'LAB':
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        elif space == 'RGB':
            return rgb_image
        elif space == 'GRAY':
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"不支持的颜色空间: {space}")
    
    def calculate_vegetation_indices(self, image):
        """
        计算植被指数
        
        Args:
            image: 输入图像（RGB格式）
        
        Returns:
            tuple: (ExG图像, CIVE图像)
        """
        # 验证图像
        valid, error_msg = validate_image(image)
        if not valid:
            raise ValueError(f"无效图像: {error_msg}")
        
        # 确保RGB格式
        rgb_image = ensure_rgb(image)
        
        # 分离通道
        r, g, b = cv2.split(rgb_image.astype(np.float32))
        
        # 计算ExG（超绿指数）: 2G-R-B
        exg = 2 * g - r - b
        exg = normalize_image(exg)
        
        # 计算CIVE（颜色不变指数）
        # CIVE = 0.441*R - 0.811*G + 0.385*B + 18.78745
        cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
        cive = normalize_image(cive)
        
        return exg, cive
    
    def apply_clahe(self, image, clip_limit=None, tile_grid_size=None):
        """
        自适应直方图均衡化(CLAHE)
        
        Args:
            image: 输入图像（RGB格式）
            clip_limit: 对比度限制，默认使用配置值
            tile_grid_size: 瓦片大小，默认使用配置值
        
        Returns:
            np.ndarray: CLAHE增强后的图像
        """
        # 验证图像
        valid, error_msg = validate_image(image)
        if not valid:
            raise ValueError(f"无效图像: {error_msg}")
        
        # 确保RGB格式
        rgb_image = ensure_rgb(image)
        
        # 使用配置值或默认值
        clip_limit = clip_limit or self.config.clahe_clip_limit
        tile_grid_size = tile_grid_size or self.config.clahe_tile_grid_size
        
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 应用CLAHE到L通道
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        # 合并通道并转换回RGB
        clahe_lab = cv2.merge((cl, a, b))
        clahe_rgb = cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2RGB)
        
        return clahe_rgb
    
    def hsv_color_thresholding(self, image, lower=None, upper=None):
        """
        HSV颜色阈值分割
        
        Args:
            image: 输入图像（RGB格式）
            lower: 下限阈值，默认使用配置的绿色下限
            upper: 上限阈值，默认使用配置的绿色上限
        
        Returns:
            np.ndarray: 二值化图像
        """
        # 验证图像
        valid, error_msg = validate_image(image)
        if not valid:
            raise ValueError(f"无效图像: {error_msg}")
        
        # 确保RGB格式
        rgb_image = ensure_rgb(image)
        
        # 转换到HSV颜色空间
        hsv = self.color_space_conversion(rgb_image, 'HSV')
        
        # 使用配置值或默认值
        lower = lower or self.config.hsv_green_lower
        upper = upper or self.config.hsv_green_upper
        
        # 颜色阈值分割
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        return mask