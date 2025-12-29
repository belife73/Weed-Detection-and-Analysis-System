"""
特征分析模块
包含几何特征分析功能
"""

import cv2
import numpy as np
from .config import Config

class FeatureAnalysis:
    """
    几何特征分析模块
    """
    
    def __init__(self, config=None):
        """
        初始化特征分析模块
        
        Args:
            config: 配置对象，包含轮廓过滤等参数
        """
        self.config = config or Config()
    
    def calculate_shape_descriptors(self, contour):
        """
        计算形状描述符
        
        Args:
            contour: 输入轮廓
        
        Returns:
            dict: 形状描述符字典，包含面积、周长、圆度、长宽比、周长面积比
        """
        if not isinstance(contour, np.ndarray):
            raise ValueError("输入必须是NumPy数组格式的轮廓")
        
        # 计算面积
        area = cv2.contourArea(contour)
        
        # 计算周长
        perimeter = cv2.arcLength(contour, True)
        
        # 计算圆度 (Circularity = 4πA/P²)
        if perimeter == 0:
            circularity = 0.0
        else:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算长宽比
        if h == 0:
            aspect_ratio = 0.0
        else:
            aspect_ratio = float(w) / h
        
        # 计算周长面积比
        if area == 0:
            perimeter_area_ratio = 0.0
        else:
            perimeter_area_ratio = perimeter / area
        
        # 计算凸包和凸缺陷
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # 计算凸度 (Solidity = A/HullA)
        if hull_area == 0:
            solidity = 0.0
        else:
            solidity = float(area) / hull_area
        
        # 计算最小外接圆
        (x_circle, y_circle), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        
        # 计算圆形度 (Compactness = A/CircleA)
        if circle_area == 0:
            compactness = 0.0
        else:
            compactness = float(area) / circle_area
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'perimeter_area_ratio': perimeter_area_ratio,
            'solidity': solidity,
            'compactness': compactness,
            'bounding_box': (x, y, w, h),
            'min_enclosing_circle': ((x_circle, y_circle), radius)
        }
    
    def filter_contours_by_area(self, contours, min_area=None, max_area=None):
        """
        根据面积过滤轮廓
        
        Args:
            contours: 轮廓列表
            min_area: 最小面积阈值，默认使用配置值
            max_area: 最大面积阈值，默认使用配置值
        
        Returns:
            list: 过滤后的轮廓列表
        """
        if not isinstance(contours, list):
            raise ValueError("输入必须是轮廓列表")
        
        # 使用配置值或默认值
        min_area = min_area or self.config.min_contour_area
        max_area = max_area or self.config.max_contour_area
        
        # 过滤轮廓
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def calculate_centroids(self, contours):
        """
        计算轮廓的质心
        
        Args:
            contours: 轮廓列表
        
        Returns:
            list: 质心坐标列表 [(cx1, cy1), (cx2, cy2), ...]
        """
        if not isinstance(contours, list):
            raise ValueError("输入必须是轮廓列表")
        
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
            else:
                # 对于面积为0的轮廓，使用边界框中心
                x, y, w, h = cv2.boundingRect(contour)
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                centroids.append((cx, cy))
        
        return centroids
    
    def classify_weed_type(self, shape_descriptors):
        """
        根据形状描述符分类杂草类型
        
        Args:
            shape_descriptors: 形状描述符字典
        
        Returns:
            str: 杂草类型，'broadleaf'（阔叶）或'grass'（禾本科）
        """
        # 基于圆度和长宽比进行分类
        # 阔叶杂草通常较圆（圆度高），长宽比接近1
        # 禾本科杂草通常狭长（圆度低），长宽比大
        
        circularity = shape_descriptors['circularity']
        aspect_ratio = shape_descriptors['aspect_ratio']
        
        if circularity > 0.5 or aspect_ratio < 1.5:
            return 'broadleaf'  # 阔叶杂草
        else:
            return 'grass'  # 禾本科杂草
    
    def calculate_contour_orientation(self, contour):
        """
        计算轮廓的方向
        
        Args:
            contour: 输入轮廓
        
        Returns:
            float: 轮廓的方向（角度）
        """
        if not isinstance(contour, np.ndarray):
            raise ValueError("输入必须是NumPy数组格式的轮廓")
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        
        # 获取旋转角度
        angle = rect[2]
        
        return angle