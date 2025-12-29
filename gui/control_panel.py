"""
控制面板组件
用于控制图像处理的各种参数和操作
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np

class ControlPanel:
    """
    控制面板类
    用于控制图像处理的各种参数和操作
    """
    
    def __init__(self, parent, on_open_image, on_process_image, on_batch_process, on_export_results, on_reset):
        """
        初始化控制面板
        
        Args:
            parent: 父窗口组件
            on_open_image: 打开图像回调函数
            on_process_image: 处理图像回调函数
            on_batch_process: 批量处理回调函数
            on_export_results: 导出结果回调函数
            on_reset: 重置回调函数
        """
        self.parent = parent
        self.on_open_image = on_open_image
        self.on_process_image = on_process_image
        self.on_batch_process = on_batch_process
        self.on_export_results = on_export_results
        self.on_reset = on_reset
        
        # 创建控制面板框架
        self.frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        
        # 创建按钮区域
        self.button_frame = ttk.Frame(self.frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        # 打开图像按钮
        self.open_btn = ttk.Button(self.button_frame, text="打开图像", command=self.open_image)
        self.open_btn.pack(fill=tk.X, pady=2)
        
        # 处理图像按钮
        self.process_btn = ttk.Button(self.button_frame, text="处理图像", command=self.process_image)
        self.process_btn.pack(fill=tk.X, pady=2)
        
        # 批量处理按钮
        self.batch_btn = ttk.Button(self.button_frame, text="批量处理", command=self.batch_process)
        self.batch_btn.pack(fill=tk.X, pady=2)
        
        # 导出结果按钮
        self.export_btn = ttk.Button(self.button_frame, text="导出结果", command=self.export_results)
        self.export_btn.pack(fill=tk.X, pady=2)
        
        # 重置按钮
        self.reset_btn = ttk.Button(self.button_frame, text="重置", command=self.reset)
        self.reset_btn.pack(fill=tk.X, pady=2)
        
        # 创建参数设置区域
        self.params_frame = ttk.LabelFrame(self.frame, text="参数设置", padding="5")
        self.params_frame.pack(fill=tk.X, pady=5)
        
        # CLAHE参数
        self.clahe_frame = ttk.LabelFrame(self.params_frame, text="CLAHE增强", padding="5")
        self.clahe_frame.pack(fill=tk.X, pady=2)
        
        # CLAHE clip limit
        self.clip_limit_label = ttk.Label(self.clahe_frame, text="对比度限制:")
        self.clip_limit_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.clip_limit_var = tk.DoubleVar(value=2.0)
        self.clip_limit_spinbox = ttk.Spinbox(self.clahe_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.clip_limit_var)
        self.clip_limit_spinbox.grid(row=0, column=1, padx=5, pady=2)
        
        # CLAHE tile grid size
        self.tile_size_label = ttk.Label(self.clahe_frame, text="瓦片大小:")
        self.tile_size_label.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.tile_size_var = tk.IntVar(value=8)
        self.tile_size_spinbox = ttk.Spinbox(self.clahe_frame, from_=1, to=32, increment=1, textvariable=self.tile_size_var)
        self.tile_size_spinbox.grid(row=1, column=1, padx=5, pady=2)
        
        # 形态学操作参数
        self.morph_frame = ttk.LabelFrame(self.params_frame, text="形态学操作", padding="5")
        self.morph_frame.pack(fill=tk.X, pady=2)
        
        # 形态学核大小
        self.kernel_size_label = ttk.Label(self.morph_frame, text="核大小:")
        self.kernel_size_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.kernel_size_var = tk.IntVar(value=5)
        self.kernel_size_spinbox = ttk.Spinbox(self.morph_frame, from_=1, to=21, increment=2, textvariable=self.kernel_size_var)
        self.kernel_size_spinbox.grid(row=0, column=1, padx=5, pady=2)
        
        # 腐蚀迭代次数
        self.erode_iter_label = ttk.Label(self.morph_frame, text="腐蚀次数:")
        self.erode_iter_label.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.erode_iter_var = tk.IntVar(value=1)
        self.erode_iter_spinbox = ttk.Spinbox(self.morph_frame, from_=0, to=5, increment=1, textvariable=self.erode_iter_var)
        self.erode_iter_spinbox.grid(row=1, column=1, padx=5, pady=2)
        
        # 膨胀迭代次数
        self.dilate_iter_label = ttk.Label(self.morph_frame, text="膨胀次数:")
        self.dilate_iter_label.grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.dilate_iter_var = tk.IntVar(value=1)
        self.dilate_iter_spinbox = ttk.Spinbox(self.morph_frame, from_=0, to=5, increment=1, textvariable=self.dilate_iter_var)
        self.dilate_iter_spinbox.grid(row=2, column=1, padx=5, pady=2)
        
        # 轮廓过滤参数
        self.contour_frame = ttk.LabelFrame(self.params_frame, text="轮廓过滤", padding="5")
        self.contour_frame.pack(fill=tk.X, pady=2)
        
        # 最小轮廓面积
        self.min_area_label = ttk.Label(self.contour_frame, text="最小面积:")
        self.min_area_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.min_area_var = tk.IntVar(value=100)
        self.min_area_spinbox = ttk.Spinbox(self.contour_frame, from_=1, to=10000, increment=10, textvariable=self.min_area_var)
        self.min_area_spinbox.grid(row=0, column=1, padx=5, pady=2)
        
        # 最大轮廓面积
        self.max_area_label = ttk.Label(self.contour_frame, text="最大面积:")
        self.max_area_label.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.max_area_var = tk.IntVar(value=100000)
        self.max_area_spinbox = ttk.Spinbox(self.contour_frame, from_=1000, to=1000000, increment=1000, textvariable=self.max_area_var)
        self.max_area_spinbox.grid(row=1, column=1, padx=5, pady=2)
    
    def open_image(self):
        """
        打开图像文件
        """
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                # 读取图像
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"无法读取图像文件: {file_path}")
                
                # 转换为RGB格式
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 调用回调函数
                self.on_open_image(image_rgb, file_path)
                
            except Exception as e:
                messagebox.showerror("错误", f"打开图像失败: {str(e)}")
    
    def process_image(self):
        """
        处理图像
        """
        # 获取参数
        params = {
            'clahe_clip_limit': self.clip_limit_var.get(),
            'clahe_tile_grid_size': (self.tile_size_var.get(), self.tile_size_var.get()),
            'morph_kernel_size': (self.kernel_size_var.get(), self.kernel_size_var.get()),
            'morph_iterations_erode': self.erode_iter_var.get(),
            'morph_iterations_dilate': self.dilate_iter_var.get(),
            'min_contour_area': self.min_area_var.get(),
            'max_contour_area': self.max_area_var.get()
        }
        
        # 调用回调函数
        self.on_process_image(params)
    
    def batch_process(self):
        """
        批量处理图像
        """
        # 调用回调函数
        self.on_batch_process()
    
    def export_results(self):
        """
        导出结果
        """
        # 调用回调函数
        self.on_export_results()
    
    def reset(self):
        """
        重置系统
        """
        # 确认重置
        if messagebox.askyesno("确认", "确定要重置系统吗？所有未保存的结果将丢失。"):
            # 重置参数
            self.clip_limit_var.set(2.0)
            self.tile_size_var.set(8)
            self.kernel_size_var.set(5)
            self.erode_iter_var.set(1)
            self.dilate_iter_var.set(1)
            self.min_area_var.set(100)
            self.max_area_var.set(100000)
            
            # 调用回调函数
            self.on_reset()
    
    def get_widget(self):
        """
        获取控制面板的主组件
        
        Returns:
            ttk.LabelFrame: 控制面板的主框架
        """
        return self.frame
    
    def get_params(self):
        """
        获取当前参数设置
        
        Returns:
            dict: 参数字典
        """
        return {
            'clahe_clip_limit': self.clip_limit_var.get(),
            'clahe_tile_grid_size': (self.tile_size_var.get(), self.tile_size_var.get()),
            'morph_kernel_size': (self.kernel_size_var.get(), self.kernel_size_var.get()),
            'morph_iterations_erode': self.erode_iter_var.get(),
            'morph_iterations_dilate': self.dilate_iter_var.get(),
            'min_contour_area': self.min_area_var.get(),
            'max_contour_area': self.max_area_var.get()
        }
    
    def set_params(self, params):
        """
        设置参数
        
        Args:
            params: 参数字典
        """
        if 'clahe_clip_limit' in params:
            self.clip_limit_var.set(params['clahe_clip_limit'])
        
        if 'clahe_tile_grid_size' in params:
            size = params['clahe_tile_grid_size']
            self.tile_size_var.set(size[0])
        
        if 'morph_kernel_size' in params:
            size = params['morph_kernel_size']
            self.kernel_size_var.set(size[0])
        
        if 'morph_iterations_erode' in params:
            self.erode_iter_var.set(params['morph_iterations_erode'])
        
        if 'morph_iterations_dilate' in params:
            self.dilate_iter_var.set(params['morph_iterations_dilate'])
        
        if 'min_contour_area' in params:
            self.min_area_var.set(params['min_contour_area'])
        
        if 'max_contour_area' in params:
            self.max_area_var.set(params['max_contour_area'])