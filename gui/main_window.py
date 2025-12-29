"""
主窗口组件
整合所有GUI组件，实现系统的主要功能和交互逻辑
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import os
import glob
from core import WeedDetector, Config
from .image_viewer import ImageViewer
from .control_panel import ControlPanel
from .result_panel import ResultPanel
from .batch_process_dialog import BatchProcessDialog

class MainWindow:
    """
    主窗口类
    整合所有GUI组件，实现系统的主要功能和交互逻辑
    """
    
    def __init__(self):
        """
        初始化主窗口
        """
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("杂草识别与分析系统")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # 设置窗口图标（如果有）
        # self.root.iconbitmap("icon.ico")
        
        # 初始化配置
        self.config = Config()
        
        # 初始化杂草检测器
        self.weed_detector = WeedDetector(self.config)
        
        # 当前打开的图像路径
        self.current_image_path = ""
        
        # 创建GUI组件
        self.create_widgets()
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_widgets(self):
        """
        创建GUI组件
        """
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧面板（控制面板和结果面板）
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 创建控制面板
        self.control_panel = ControlPanel(
            left_panel,
            on_open_image=self.on_open_image,
            on_process_image=self.on_process_image,
            on_batch_process=self.on_batch_process,
            on_export_results=self.on_export_results,
            on_reset=self.on_reset
        )
        self.control_panel.get_widget().pack(fill=tk.X, pady=5)
        
        # 创建结果面板
        self.result_panel = ResultPanel(left_panel)
        self.result_panel.get_widget().pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建图像查看器
        self.image_viewer = ImageViewer(main_frame)
        self.image_viewer.get_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建菜单栏（在组件创建之后）
        self.create_menu()
    
    def create_menu(self):
        """
        创建菜单栏
        """
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开图像", command=self.control_panel.open_image)
        file_menu.add_command(label="批量处理", command=self.on_batch_process)
        file_menu.add_separator()
        file_menu.add_command(label="导出结果", command=self.on_export_results)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_close)
        
        # 处理菜单
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="处理", menu=process_menu)
        process_menu.add_command(label="处理图像", command=self.on_process_image)
        process_menu.add_separator()
        process_menu.add_command(label="重置", command=self.on_reset)
        
        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)
    
    def on_open_image(self, image, file_path):
        """
        打开图像回调函数
        
        Args:
            image: 打开的图像（RGB格式）
            file_path: 图像文件路径
        """
        self.current_image_path = file_path
        self.image_viewer.display_original_image(image)
        self.result_panel.clear_results()
        
        # 更新窗口标题
        self.root.title(f"杂草识别与分析系统 - {os.path.basename(file_path)}")
    
    def on_process_image(self, params=None):
        """
        处理图像回调函数
        
        Args:
            params: 处理参数
        """
        if self.image_viewer.original_image is None:
            messagebox.showwarning("警告", "请先打开一张图像")
            return
        
        try:
            # 更新配置参数
            if params:
                self.config.update(**params)
                
            # 更新检测器配置
            self.weed_detector = WeedDetector(self.config)
            
            # 处理图像
            results = self.weed_detector.process_single_image(self.image_viewer.original_image)
            
            # 生成标注图像
            annotated_image = self.weed_detector.annotate_results()
            
            # 显示结果
            self.image_viewer.display_processed_image(annotated_image)
            self.result_panel.update_results(results)
            
        except Exception as e:
            messagebox.showerror("错误", f"图像处理失败: {str(e)}")
    
    def on_batch_process(self):
        """
        批量处理回调函数
        """
        # 创建批量处理对话框
        dialog = BatchProcessDialog(self.root, self.batch_process_callback)
        dialog.show()
    
    def batch_process_callback(self, input_folder, output_folder, progress_callback):
        """
        批量处理回调函数
        
        Args:
            input_folder: 输入文件夹
            output_folder: 输出文件夹
            progress_callback: 进度回调函数
        
        Returns:
            list: 处理结果列表
        """
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        
        total_files = len(image_files)
        if total_files == 0:
            raise ValueError("输入文件夹中没有图像文件")
        
        results_list = []
        
        # 处理每张图像
        for i, image_path in enumerate(image_files):
            # 检查是否需要停止
            if not progress_callback(i + 1, total_files, os.path.basename(image_path)):
                break
            
            try:
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # 转换为RGB格式
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 创建新的检测器实例，确保配置更新
                detector = WeedDetector(self.config)
                
                # 处理图像
                results = detector.process_single_image(image_rgb)
                
                # 生成标注图像
                annotated_image = detector.annotate_results()
                
                # 保存标注图像
                output_image_name = self.config.batch_output_format.format(os.path.basename(image_path))
                output_image_path = os.path.join(output_folder, output_image_name)
                cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                
                # 保存结果
                results['image_name'] = os.path.basename(image_path)
                results_list.append(results)
                
            except Exception as e:
                continue
        
        return results_list
    
    def on_export_results(self):
        """
        导出结果回调函数
        """
        if not self.weed_detector.results:
            messagebox.showwarning("警告", "没有可导出的结果")
            return
        
        # 选择保存位置
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV文件", "*.csv"),
                ("所有文件", "*.*")
            ],
            title="导出结果"
        )
        
        if not file_path:
            return
        
        try:
            # 准备数据
            results = self.weed_detector.results
            
            # 创建数据字典
            data = {
                'weed_count': results.get('weed_count', 0),
                'coverage': results.get('coverage', 0.0),
                'mean_circularity': results.get('shape_stats', {}).get('mean_circularity', 0.0),
                'mean_aspect_ratio': results.get('shape_stats', {}).get('mean_aspect_ratio', 0.0),
                'mean_perimeter_area_ratio': results.get('shape_stats', {}).get('mean_perimeter_area_ratio', 0.0)
            }
            
            # 添加颜色矩数据
            color_moments = results.get('color_moments', {})
            for channel, moments in color_moments.items():
                data[f'{channel}_mean'] = moments.get('mean', 0.0)
                data[f'{channel}_variance'] = moments.get('variance', 0.0)
            
            # 添加纹理特征数据
            glcm_features = results.get('glcm_features', {})
            for feature, value in glcm_features.items():
                data[f'glcm_{feature}'] = value
            
            # 转换为DataFrame并导出
            import pandas as pd
            df = pd.DataFrame([data])
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            messagebox.showinfo("信息", f"结果已成功导出到: {file_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出结果失败: {str(e)}")
    
    def on_reset(self):
        """
        重置回调函数
        """
        # 重置图像查看器
        self.image_viewer.clear_images()
        
        # 重置结果面板
        self.result_panel.clear_results()
        
        # 重置检测器
        self.weed_detector.reset()
        
        # 重置窗口标题
        self.root.title("杂草识别与分析系统")
        
        # 重置当前图像路径
        self.current_image_path = ""
    
    def show_about(self):
        """
        显示关于对话框
        """
        about_text = """杂草识别与分析系统 v1.0

基于传统计算机视觉算法的杂草识别与分析系统，
包含图像增强、分割、特征提取和统计分析等功能。

© 2025 杂草识别系统开发团队"""
        
        messagebox.showinfo("关于", about_text)
    
    def on_close(self):
        """
        窗口关闭事件处理
        """
        if messagebox.askyesno("确认", "确定要退出系统吗？"):
            self.root.destroy()
    
    def run(self):
        """
        运行主窗口
        """
        self.root.mainloop()

# 测试代码
if __name__ == "__main__":
    app = MainWindow()
    app.run()