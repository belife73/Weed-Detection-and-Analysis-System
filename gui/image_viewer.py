"""
图像查看器组件
用于显示原始图像和处理结果图像
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageViewer:
    """
    图像查看器类
    用于显示原始图像和处理结果图像
    """
    
    def __init__(self, parent):
        """
        初始化图像查看器
        
        Args:
            parent: 父窗口组件
        """
        self.parent = parent
        self.original_image = None
        self.processed_image = None
        self.original_photo = None
        self.processed_photo = None
        
        # 创建图像显示框架
        self.frame = ttk.LabelFrame(parent, text="图像显示", padding="10")
        
        # 创建原始图像标签
        self.original_label = ttk.Label(self.frame, text="原始图像")
        self.original_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.N)
        
        self.original_canvas = tk.Canvas(self.frame, width=500, height=500, bg="gray")
        self.original_canvas.grid(row=1, column=0, padx=5, pady=5)
        
        # 创建处理后图像标签
        self.processed_label = ttk.Label(self.frame, text="处理结果")
        self.processed_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.N)
        
        self.processed_canvas = tk.Canvas(self.frame, width=500, height=500, bg="gray")
        self.processed_canvas.grid(row=1, column=1, padx=5, pady=5)
        
        # 创建缩放控制
        self.zoom_frame = ttk.Frame(self.frame)
        self.zoom_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        self.zoom_label = ttk.Label(self.zoom_frame, text="缩放:")
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        self.zoom_scale = ttk.Scale(self.zoom_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL, length=300)
        self.zoom_scale.set(1.0)
        self.zoom_scale.pack(side=tk.LEFT, padx=5)
        
        self.zoom_value = ttk.Label(self.zoom_frame, text="100%")
        self.zoom_value.pack(side=tk.LEFT, padx=5)
        
        # 绑定缩放事件
        self.zoom_scale.bind("<Motion>", self.on_zoom_changed)
        self.zoom_scale.bind("<ButtonRelease-1>", self.on_zoom_changed)
    
    def on_zoom_changed(self, event=None):
        """
        缩放比例变化事件处理
        """
        zoom = self.zoom_scale.get()
        self.zoom_value.config(text=f"{int(zoom * 100)}%")
        
        # 重新显示图像
        if self.original_image is not None:
            self.display_original_image(self.original_image)
        
        if self.processed_image is not None:
            self.display_processed_image(self.processed_image)
    
    def display_original_image(self, image):
        """
        显示原始图像
        
        Args:
            image: 原始图像（RGB格式）
        """
        self.original_image = image
        self._display_image(image, self.original_canvas, "original")
    
    def display_processed_image(self, image):
        """
        显示处理后的图像
        
        Args:
            image: 处理后的图像（RGB格式）
        """
        self.processed_image = image
        self._display_image(image, self.processed_canvas, "processed")
    
    def _display_image(self, image, canvas, image_type):
        """
        内部方法：在画布上显示图像
        
        Args:
            image: 要显示的图像
            canvas: 目标画布
            image_type: 图像类型（"original"或"processed"）
        """
        if image is None:
            return
        
        # 获取缩放比例
        zoom = self.zoom_scale.get()
        
        # 调整图像大小
        h, w, _ = image.shape
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h))
        
        # 转换为PIL图像
        pil_image = Image.fromarray(resized)
        
        # 创建PhotoImage对象
        photo = ImageTk.PhotoImage(pil_image)
        
        # 清除画布
        canvas.delete("all")
        
        # 显示图像
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
        # 保存PhotoImage引用，防止被垃圾回收
        if image_type == "original":
            self.original_photo = photo
        else:
            self.processed_photo = photo
        
        # 更新画布大小
        canvas.config(width=new_w, height=new_h)
    
    def clear_images(self):
        """
        清除显示的图像
        """
        self.original_image = None
        self.processed_image = None
        self.original_photo = None
        self.processed_photo = None
        
        self.original_canvas.delete("all")
        self.processed_canvas.delete("all")
        
        # 重置画布背景
        self.original_canvas.config(width=500, height=500, bg="gray")
        self.processed_canvas.config(width=500, height=500, bg="gray")
    
    def get_widget(self):
        """
        获取图像查看器的主组件
        
        Returns:
            ttk.LabelFrame: 图像查看器的主框架
        """
        return self.frame