"""
结果面板组件
用于显示图像处理的结果信息
"""

import tkinter as tk
from tkinter import ttk
import numpy as np

class ResultPanel:
    """
    结果面板类
    用于显示图像处理的结果信息
    """
    
    def __init__(self, parent):
        """
        初始化结果面板
        
        Args:
            parent: 父窗口组件
        """
        self.parent = parent
        
        # 创建结果面板框架
        self.frame = ttk.LabelFrame(parent, text="分析结果", padding="10")
        
        # 创建结果文本框
        self.result_text = tk.Text(
            self.frame, 
            width=40, 
            height=20, 
            wrap=tk.WORD,
            font=(
                "Arial", 
                10
            )
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建滚动条
        self.scrollbar = ttk.Scrollbar(
            self.frame, 
            orient=tk.VERTICAL, 
            command=self.result_text.yview
        )
        self.result_text.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建复制按钮
        self.copy_btn = ttk.Button(
            self.frame, 
            text="复制结果", 
            command=self.copy_results
        )
        self.copy_btn.pack(fill=tk.X, pady=5)
    
    def update_results(self, results):
        """
        更新结果显示
        
        Args:
            results: 处理结果字典
        """
        # 清空文本框
        self.result_text.delete(1.0, tk.END)
        
        if not results:
            self.result_text.insert(tk.END, "尚未处理图像，没有结果可显示。")
            return
        
        # 格式化结果
        formatted_results = self._format_results(results)
        
        # 插入结果
        self.result_text.insert(tk.END, formatted_results)
        
        # 滚动到顶部
        self.result_text.see(tk.END)
    
    def _format_results(self, results):
        """
        格式化结果文本
        
        Args:
            results: 处理结果字典
        
        Returns:
            str: 格式化后的结果文本
        """
        text = "# 杂草分析结果\n"
        text += "=" * 30 + "\n\n"
        
        # 基本信息
        text += "## 基本信息\n"
        text += f"- 杂草数量: {results.get('weed_count', 0)}\n"
        text += f"- 覆盖度: {results.get('coverage', 0.0):.2f}%\n\n"
        
        # 形状描述符统计
        shape_stats = results.get('shape_stats', {})
        if shape_stats:
            text += "## 形状描述符统计\n"
            text += f"- 平均面积: {shape_stats.get('mean_area', 0.0):.2f} 像素\n"
            text += f"- 平均圆度: {shape_stats.get('mean_circularity', 0.0):.4f}\n"
            text += f"- 平均长宽比: {shape_stats.get('mean_aspect_ratio', 0.0):.4f}\n"
            text += f"- 平均周长面积比: {shape_stats.get('mean_perimeter_area_ratio', 0.0):.4f}\n"
            text += f"- 面积标准差: {shape_stats.get('std_area', 0.0):.2f}\n"
            text += f"- 圆度标准差: {shape_stats.get('std_circularity', 0.0):.4f}\n\n"
        
        # 颜色矩分析
        color_moments = results.get('color_moments', {})
        if color_moments:
            text += "## 颜色矩分析\n"
            for channel, moments in color_moments.items():
                text += f"### {channel}通道\n"
                text += f"- 均值: {moments.get('mean', 0.0):.2f}\n"
                text += f"- 方差: {moments.get('variance', 0.0):.2f}\n"
                text += f"- 偏度: {moments.get('skewness', 0.0):.4f}\n"
                text += f"- 峰度: {moments.get('kurtosis', 0.0):.4f}\n\n"
        
        # 纹理特征
        glcm_features = results.get('glcm_features', {})
        if glcm_features:
            text += "## 纹理特征\n"
            text += f"- 对比度: {glcm_features.get('contrast', 0.0):.4f}\n"
            text += f"- 相异性: {glcm_features.get('dissimilarity', 0.0):.4f}\n"
            text += f"- 均匀性: {glcm_features.get('homogeneity', 0.0):.4f}\n"
            text += f"- 能量: {glcm_features.get('energy', 0.0):.4f}\n"
            text += f"- 相关性: {glcm_features.get('correlation', 0.0):.4f}\n\n"
        
        # 形状描述符详细信息
        shape_descriptors = results.get('shape_descriptors', [])
        if shape_descriptors:
            text += "## 形状描述符详细信息\n"
            for i, desc in enumerate(shape_descriptors[:5]):  # 只显示前5个
                text += f"### 杂草 {i+1}\n"
                text += f"- 面积: {desc.get('area', 0.0):.2f} 像素\n"
                text += f"- 周长: {desc.get('perimeter', 0.0):.2f} 像素\n"
                text += f"- 圆度: {desc.get('circularity', 0.0):.4f}\n"
                text += f"- 长宽比: {desc.get('aspect_ratio', 0.0):.4f}\n"
                text += f"- 周长面积比: {desc.get('perimeter_area_ratio', 0.0):.4f}\n"
                text += f"- 凸度: {desc.get('solidity', 0.0):.4f}\n"
                text += f"- 圆形度: {desc.get('compactness', 0.0):.4f}\n\n"
            
            if len(shape_descriptors) > 5:
                text += f"... 还有 {len(shape_descriptors) - 5} 个杂草的详细信息未显示\n\n"
        
        return text
    
    def copy_results(self):
        """
        复制结果到剪贴板
        """
        # 获取文本内容
        content = self.result_text.get(1.0, tk.END)
        
        # 复制到剪贴板
        self.parent.clipboard_clear()
        self.parent.clipboard_append(content)
        
        # 显示复制成功信息
        self.parent.update()  # 更新剪贴板
    
    def clear_results(self):
        """
        清空结果显示
        """
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "尚未处理图像，没有结果可显示。")
    
    def get_widget(self):
        """
        获取结果面板的主组件
        
        Returns:
            ttk.LabelFrame: 结果面板的主框架
        """
        return self.frame