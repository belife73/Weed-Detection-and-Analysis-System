"""
批量处理对话框组件
用于设置和执行批量处理操作
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import time

class BatchProcessDialog:
    """
    批量处理对话框类
    用于设置和执行批量处理操作
    """
    
    def __init__(self, parent, batch_process_callback):
        """
        初始化批量处理对话框
        
        Args:
            parent: 父窗口组件
            batch_process_callback: 批量处理回调函数
        """
        self.parent = parent
        self.batch_process_callback = batch_process_callback
        self.dialog = None
        self.input_folder = ""
        self.output_folder = ""
        self.is_processing = False
        
    def show(self):
        """
        显示批量处理对话框
        """
        # 创建对话框
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("批量处理")
        self.dialog.geometry("500x300")
        self.dialog.resizable(False, False)
        
        # 设置对话框居中
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # 创建主框架
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 输入文件夹选择
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        input_label = ttk.Label(input_frame, text="输入文件夹:")
        input_label.pack(side=tk.LEFT, padx=5)
        
        self.input_entry = ttk.Entry(input_frame)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        input_btn = ttk.Button(input_frame, text="浏览", command=self.select_input_folder)
        input_btn.pack(side=tk.LEFT, padx=5)
        
        # 输出文件夹选择
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        output_label = ttk.Label(output_frame, text="输出文件夹:")
        output_label.pack(side=tk.LEFT, padx=5)
        
        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        output_btn = ttk.Button(output_frame, text="浏览", command=self.select_output_folder)
        output_btn.pack(side=tk.LEFT, padx=5)
        
        # 进度条
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_label = ttk.Label(progress_frame, text="准备开始...")
        self.progress_label.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # 日志文本框
        log_frame = ttk.LabelFrame(main_frame, text="处理日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, font=('Arial', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建滚动条
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 开始按钮
        self.start_btn = ttk.Button(button_frame, text="开始处理", command=self.start_batch_process)
        self.start_btn.pack(side=tk.RIGHT, padx=5)
        
        # 取消按钮
        self.cancel_btn = ttk.Button(button_frame, text="取消", command=self.cancel_batch_process)
        self.cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        # 等待对话框关闭
        self.parent.wait_window(self.dialog)
    
    def select_input_folder(self):
        """
        选择输入文件夹
        """
        folder = filedialog.askdirectory(title="选择输入文件夹")
        if folder:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, folder)
            self.input_folder = folder
    
    def select_output_folder(self):
        """
        选择输出文件夹
        """
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder)
            self.output_folder = folder
    
    def start_batch_process(self):
        """
        开始批量处理
        """
        # 获取输入输出文件夹
        self.input_folder = self.input_entry.get().strip()
        self.output_folder = self.output_entry.get().strip()
        
        # 验证输入
        if not self.input_folder:
            messagebox.showerror("错误", "请选择输入文件夹")
            return
        
        if not os.path.exists(self.input_folder):
            messagebox.showerror("错误", "输入文件夹不存在")
            return
        
        if not self.output_folder:
            messagebox.showerror("错误", "请选择输出文件夹")
            return
        
        if not os.path.exists(self.output_folder):
            try:
                os.makedirs(self.output_folder)
            except Exception as e:
                messagebox.showerror("错误", f"无法创建输出文件夹: {str(e)}")
                return
        
        # 禁用按钮
        self.start_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(text="停止")
        
        # 清空日志
        self.log_text.delete(1.0, tk.END)
        
        # 开始处理线程
        self.is_processing = True
        thread = threading.Thread(target=self.run_batch_process)
        thread.daemon = True
        thread.start()
    
    def run_batch_process(self):
        """
        在后台线程中执行批量处理
        """
        try:
            # 调用批量处理回调函数
            self.log_message("开始批量处理...")
            self.progress_label.config(text="正在获取图像文件...")
            
            # 执行批量处理
            results = self.batch_process_callback(self.input_folder, self.output_folder, self.update_progress)
            
            if self.is_processing:
                self.progress_label.config(text="处理完成！")
                self.progress_bar.config(value=100)
                self.log_message(f"批量处理完成，共处理 {len(results)} 张图像")
                self.log_message(f"结果已保存到: {self.output_folder}")
                self.start_btn.config(state=tk.NORMAL)
                self.cancel_btn.config(text="关闭")
        
        except Exception as e:
            if self.is_processing:
                self.log_message(f"处理错误: {str(e)}")
                self.progress_label.config(text="处理出错！")
                self.start_btn.config(state=tk.NORMAL)
                self.cancel_btn.config(text="关闭")
    
    def update_progress(self, current, total, filename):
        """
        更新进度
        
        Args:
            current: 当前处理数量
            total: 总数量
            filename: 当前处理的文件名
        """
        if not self.is_processing:
            return False
        
        # 计算进度百分比
        progress = int((current / total) * 100)
        
        # 更新UI
        self.dialog.after(0, lambda: self.progress_bar.config(value=progress))
        self.dialog.after(0, lambda: self.progress_label.config(text=f"正在处理: {current}/{total}"))
        self.dialog.after(0, lambda: self.log_message(f"处理中: {filename}"))
        
        return True
    
    def log_message(self, message):
        """
        记录日志消息
        
        Args:
            message: 日志消息
        """
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.dialog.after(0, lambda: self.log_text.insert(tk.END, log_entry))
        self.dialog.after(0, lambda: self.log_text.see(tk.END))
    
    def cancel_batch_process(self):
        """
        取消或关闭批量处理
        """
        if self.is_processing:
            # 停止处理
            self.is_processing = False
            self.log_message("正在停止处理...")
            self.cancel_btn.config(state=tk.DISABLED)
        else:
            # 关闭对话框
            self.dialog.destroy()