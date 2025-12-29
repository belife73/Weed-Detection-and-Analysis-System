"""
图像分割模块
包含图像分割与背景剔除功能
"""

import cv2
import numpy as np
from .utils import validate_image
from .config import Config

class Segmentation:
    """
    图像分割与背景剔除模块
    """
    
    def __init__(self, config=None):
        """
        初始化分割模块
        
        Args:
            config: 配置对象，包含形态学操作等参数
        """
        self.config = config or Config()
    
    def otsu_thresholding(self, image, channel=0):
        """
        大津法自动阈值分割
        
        Args:
            image: 输入图像（灰度或彩色）
            channel: 彩色图像时使用的通道索引
        
        Returns:
            np.ndarray: 二值化图像
        """
        # 验证图像
        valid, error_msg = validate_image(image)
        if not valid:
            raise ValueError(f"无效图像: {error_msg}")
        
        # 处理彩色图像
        if len(image.shape) == 3:
            gray = image[:, :, channel]
        else:
            gray = image
        
        # 应用大津法阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def morphological_operations(self, binary_image, kernel_size=None, iterations_erode=None, iterations_dilate=None):
        """
        形态学操作（腐蚀+膨胀）
        
        Args:
            binary_image: 二值化输入图像
            kernel_size: 结构元素大小，默认使用配置值
            iterations_erode: 腐蚀迭代次数，默认使用配置值
            iterations_dilate: 膨胀迭代次数，默认使用配置值
        
        Returns:
            np.ndarray: 形态学处理后的二值化图像
        """
        # 验证图像
        valid, error_msg = validate_image(binary_image)
        if not valid:
            raise ValueError(f"无效图像: {error_msg}")
        
        # 确保二值化图像
        if len(binary_image.shape) == 3:
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_RGB2GRAY)
        
        # 使用配置值或默认值
        kernel_size = kernel_size or self.config.morph_kernel_size
        iterations_erode = iterations_erode or self.config.morph_iterations_erode
        iterations_dilate = iterations_dilate or self.config.morph_iterations_dilate
        
        # 创建结构元素
        kernel = np.ones(kernel_size, np.uint8)
        
        # 腐蚀操作（去除噪点）
        eroded = cv2.erode(binary_image, kernel, iterations=iterations_erode)
        
        # 膨胀操作（填补空洞）
        dilated = cv2.dilate(eroded, kernel, iterations=iterations_dilate)
        
        return dilated
    
    def watershed_segmentation(self, image):
        """
        分水岭算法分割
        
        Args:
            image: 输入图像（RGB格式）
        
        Returns:
            tuple: (分割结果图像, 标记图像)
        """
        # 验证图像
        valid, error_msg = validate_image(image)
        if not valid:
            raise ValueError(f"无效图像: {error_msg}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 高斯模糊，减少噪点
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 计算梯度
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        
        # 阈值处理获取前景区域
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 距离变换
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # 阈值处理获取确定的前景区域
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # 膨胀操作获取确定的背景区域
        sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)
        
        # 转换数据类型
        sure_fg = np.uint8(sure_fg)
        
        # 未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记连通区域
        _, markers = cv2.connectedComponents(sure_fg)
        
        # 确保背景标记为1，前景从2开始
        markers = markers + 1
        
        # 未知区域标记为0
        markers[unknown == 255] = 0
        
        # 应用分水岭算法
        markers = cv2.watershed(image, markers)
        
        # 生成结果图像
        result = image.copy()
        
        # 边界用红色标记
        result[markers == -1] = [255, 0, 0]
        
        return result, markers
    
    def extract_contours(self, binary_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
        """
        提取图像中的轮廓
        
        Args:
            binary_image: 二值化输入图像
            mode: 轮廓检索模式
            method: 轮廓近似方法
        
        Returns:
            list: 轮廓列表
        """
        # 验证图像
        valid, error_msg = validate_image(binary_image)
        if not valid:
            raise ValueError(f"无效图像: {error_msg}")
        
        # 确保二值化图像
        if len(binary_image.shape) == 3:
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_RGB2GRAY)
        
        # 提取轮廓
        contours, _ = cv2.findContours(binary_image, mode, method)
        
        # 确保返回的是列表类型
        if isinstance(contours, np.ndarray):
            # 将numpy数组转换为列表
            return contours.tolist()
        return list(contours)