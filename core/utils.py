"""
工具模块
包含日志设置、图像验证等通用功能
"""

import logging
import os
from datetime import datetime
import cv2
import numpy as np

def setup_logger(name, log_dir, level=logging.INFO):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录
        level: 日志级别
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建日志文件名（按日期）
    log_filename = f"weed_detection_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # 定义日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def validate_image(image):
    """
    验证图像是否有效
    
    Args:
        image: 输入图像
    
    Returns:
        bool: 图像是否有效
        str: 错误信息（如果无效）
    """
    if image is None:
        return False, "图像为空"
    
    if not isinstance(image, np.ndarray):
        return False, "图像必须是NumPy数组"
    
    if len(image.shape) not in [2, 3]:
        return False, "图像必须是2D（灰度）或3D（彩色）"
    
    if image.shape[0] == 0 or image.shape[1] == 0:
        return False, "图像尺寸无效"
    
    return True, ""

def ensure_rgb(image):
    """
    确保图像是RGB格式
    
    Args:
        image: 输入图像
    
    Returns:
        np.ndarray: RGB格式的图像
    """
    valid, error_msg = validate_image(image)
    if not valid:
        raise ValueError(f"无效图像: {error_msg}")
    
    if len(image.shape) == 2:
        # 灰度图转换为RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # RGBA转换为RGB
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:
        # 已为RGB
        return image
    else:
        raise ValueError(f"不支持的图像通道数: {image.shape[2]}")

def normalize_image(image):
    """
    归一化图像到0-255范围
    
    Args:
        image: 输入图像
    
    Returns:
        np.ndarray: 归一化后的图像
    """
    valid, error_msg = validate_image(image)
    if not valid:
        raise ValueError(f"无效图像: {error_msg}")
    
    if image.dtype == np.uint8:
        return image
    
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)
    
    normalized = ((image - min_val) / (max_val - min_val)) * 255
    return normalized.astype(np.uint8)