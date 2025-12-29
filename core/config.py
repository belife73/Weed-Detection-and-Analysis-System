"""
系统配置模块
存储系统的各种参数设置
"""

import os

class Config:
    """系统配置类"""
    
    def __init__(self):
        # 图像增强参数
        self.clahe_clip_limit = 2.0
        self.clahe_tile_grid_size = (8, 8)
        
        # 形态学操作参数
        self.morph_kernel_size = (5, 5)
        self.morph_iterations_erode = 1
        self.morph_iterations_dilate = 1
        
        # 轮廓过滤参数
        self.min_contour_area = 100
        self.max_contour_area = 100000
        
        # 颜色阈值参数
        self.hsv_green_lower = (25, 40, 40)
        self.hsv_green_upper = (90, 255, 255)
        
        # 日志配置
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        self.log_level = 'INFO'
        
        # 批量处理配置
        self.batch_output_format = 'annotated_{}'
        
        # 结果导出配置
        self.result_csv_columns = [
            'image_name', 'weed_count', 'coverage', 'mean_circularity', 
            'mean_aspect_ratio', 'mean_perimeter_area_ratio'
        ]
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_config_dict(self):
        """获取配置字典"""
        return {
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_tile_grid_size': self.clahe_tile_grid_size,
            'morph_kernel_size': self.morph_kernel_size,
            'morph_iterations_erode': self.morph_iterations_erode,
            'morph_iterations_dilate': self.morph_iterations_dilate,
            'min_contour_area': self.min_contour_area,
            'max_contour_area': self.max_contour_area,
            'hsv_green_lower': self.hsv_green_lower,
            'hsv_green_upper': self.hsv_green_upper,
            'log_dir': self.log_dir,
            'log_level': self.log_level,
            'batch_output_format': self.batch_output_format,
            'result_csv_columns': self.result_csv_columns
        }