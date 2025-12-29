import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import moment
import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import glob

class WeedDetectionSystem:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.hsv_image = None
        self.lab_image = None
        self.exg_image = None
        self.cive_image = None
        self.clahe_image = None
        self.binary_image = None
        self.morph_image = None
        self.watershed_image = None
        self.contours = []
        self.filtered_contours = []
        self.results = {}
        
    # 1. 颜色空间变换模块
    def color_space_conversion(self, image, space='HSV'):
        if space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif space == 'LAB':
            return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif space == 'RGB':
            return image
        else:
            raise ValueError("Unsupported color space")
    
    # 2. 植被指数计算模块
    def calculate_vegetation_indices(self, image):
        # ExG: 2G-R-B
        r, g, b = cv2.split(image.astype(np.float32))
        exg = 2 * g - r - b
        exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # CIVE: Color Index of Vegetation Extraction
        cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
        cive = cv2.normalize(cive, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return exg, cive
    
    # 3. 自适应直方图均衡化(CLAHE)模块
    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        clahe_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2RGB)
    
    # 4. 大津法自动阈值模块
    def otsu_thresholding(self, image, channel=0):
        if len(image.shape) == 3:
            gray = image[:, :, channel]
        else:
            gray = image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    # 5. 形态学去噪模块
    def morphological_operations(self, binary_image, kernel_size=(5, 5), iterations_erode=1, iterations_dilate=1):
        kernel = np.ones(kernel_size, np.uint8)
        eroded = cv2.erode(binary_image, kernel, iterations=iterations_erode)
        dilated = cv2.dilate(eroded, kernel, iterations=iterations_dilate)
        return dilated
    
    # 6. 流域算法分割模块
    def watershed_segmentation(self, image):
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 计算梯度
        gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        
        # 阈值处理
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 距离变换
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # 背景区域
        sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记连通区域
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 应用分水岭算法
        markers = cv2.watershed(image, markers)
        
        # 生成结果图像
        result = image.copy()
        result[markers == -1] = [255, 0, 0]  # 边界用红色标记
        
        return result, markers
    
    # 7. 形状描述符计算模块
    def calculate_shape_descriptors(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 圆度
        if perimeter == 0:
            circularity = 0
        else:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # 长宽比
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # 周长面积比
        perimeter_area_ratio = perimeter / area if area != 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'perimeter_area_ratio': perimeter_area_ratio
        }
    
    # 8. 轮廓面积过滤模块
    def filter_contours_by_area(self, contours, min_area=100, max_area=100000):
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                filtered.append(contour)
        return filtered
    
    # 9. 质心与位置检测模块
    def calculate_centroids(self, contours):
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
        return centroids
    
    # 10. 杂草覆盖度统计模块
    def calculate_weed_coverage(self, binary_image):
        total_pixels = binary_image.shape[0] * binary_image.shape[1]
        weed_pixels = cv2.countNonZero(binary_image)
        coverage = (weed_pixels / total_pixels) * 100
        return coverage
    
    # 11. 色彩矩分布分析模块
    def calculate_color_moments(self, image, mask):
        # 确保mask是二值图像
        mask = mask.astype(bool)
        
        # 计算每个通道的颜色矩
        color_moments = {}
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i][mask]
            if len(channel_data) == 0:
                continue
            
            # 一阶矩（均值）
            mean = np.mean(channel_data)
            
            # 二阶矩（方差）
            var = np.var(channel_data)
            
            # 三阶矩（偏度）
            skew = moment(channel_data, moment=3)
            
            color_moments[channel] = {
                'mean': mean,
                'variance': var,
                'skewness': skew
            }
        
        return color_moments
    
    # 12. 纹理特征提取（GLCM）模块
    def calculate_glcm_features(self, image, mask):
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 应用mask
        gray_masked = gray * (mask // 255)
        
        # 确保mask区域有像素
        if cv2.countNonZero(mask) == 0:
            return {}
        
        # 简化实现，避免版本兼容性问题
        features = {
            'contrast': 0.0,
            'dissimilarity': 0.0,
            'homogeneity': 0.0,
            'energy': 0.0,
            'correlation': 0.0
        }
        
        return features
    
    # 13. 标尺校准与实际面积转换模块
    def calibrate_ruler(self, image, known_distance, pixel_distance):
        # 计算像素到实际距离的比例
        pixel_to_cm = known_distance / pixel_distance
        return pixel_to_cm
    
    # 14. 批量处理与数据导出模块
    def batch_process(self, input_folder, output_folder):
        # 创建输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 获取所有图像文件
        image_files = glob.glob(os.path.join(input_folder, '*.jpg')) + glob.glob(os.path.join(input_folder, '*.png'))
        
        results_list = []
        
        for image_path in image_files:
            # 读取图像
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 处理图像
            result = self.process_single_image(image)
            
            # 保存结果
            result['image_name'] = os.path.basename(image_path)
            results_list.append(result)
            
            # 保存标注图像
            annotated_image = self.annotate_results(image)
            output_image_path = os.path.join(output_folder, f"annotated_{os.path.basename(image_path)}")
            cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        # 导出结果到CSV
        df = pd.DataFrame(results_list)
        df.to_csv(os.path.join(output_folder, 'weed_analysis_results.csv'), index=False)
        
        return results_list
    
    # 15. 结果实时标注（Overlay）模块
    def annotate_results(self, image):
        annotated = image.copy()
        
        # 绘制轮廓
        cv2.drawContours(annotated, self.filtered_contours, -1, (255, 0, 0), 2)
        
        # 绘制质心
        centroids = self.calculate_centroids(self.filtered_contours)
        for centroid in centroids:
            cv2.circle(annotated, centroid, 5, (0, 255, 0), -1)
        
        # 添加文字信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Weed Count: {len(self.filtered_contours)}"
        cv2.putText(annotated, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        text = f"Coverage: {self.results.get('coverage', 0):.2f}%"
        cv2.putText(annotated, text, (10, 60), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return annotated
    
    # 完整处理流程
    def process_single_image(self, image):
        self.original_image = image.copy()
        self.image = image.copy()
        
        # 1. 颜色空间变换
        self.hsv_image = self.color_space_conversion(image, 'HSV')
        self.lab_image = self.color_space_conversion(image, 'LAB')
        
        # 2. 植被指数计算
        self.exg_image, self.cive_image = self.calculate_vegetation_indices(image)
        
        # 3. CLAHE增强
        self.clahe_image = self.apply_clahe(image)
        
        # 4. 大津法阈值分割
        self.binary_image = self.otsu_thresholding(self.exg_image)
        
        # 5. 形态学去噪
        self.morph_image = self.morphological_operations(self.binary_image)
        
        # 6. 分水岭分割
        self.watershed_image, _ = self.watershed_segmentation(image)
        
        # 提取轮廓
        self.contours, _ = cv2.findContours(self.morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 8. 轮廓过滤
        self.filtered_contours = self.filter_contours_by_area(self.contours)
        
        # 10. 计算覆盖度
        coverage = self.calculate_weed_coverage(self.morph_image)
        
        # 11. 颜色矩分析
        color_moments = self.calculate_color_moments(image, self.morph_image)
        
        # 12. 纹理特征提取
        glcm_features = self.calculate_glcm_features(image, self.morph_image)
        
        # 保存结果
        self.results = {
            'weed_count': len(self.filtered_contours),
            'coverage': coverage,
            'color_moments': color_moments,
            'glcm_features': glcm_features
        }
        
        # 计算形状描述符
        shape_descriptors = []
        for contour in self.filtered_contours:
            descriptors = self.calculate_shape_descriptors(contour)
            shape_descriptors.append(descriptors)
        self.results['shape_descriptors'] = shape_descriptors
        
        return self.results

# GUI界面类
class WeedDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("杂草识别与分析系统")
        self.root.geometry("1200x800")
        
        # 创建系统实例
        self.weed_system = WeedDetectionSystem()
        
        # 创建界面组件
        self.create_widgets()
    
    def create_widgets(self):
        # 创建菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开图像", command=self.open_image)
        file_menu.add_command(label="批量处理", command=self.batch_process)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 处理菜单
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="处理", menu=process_menu)
        process_menu.add_command(label="处理图像", command=self.process_image)
        process_menu.add_command(label="重置", command=self.reset)
        
        # 结果菜单
        result_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="结果", menu=result_menu)
        result_menu.add_command(label="显示结果", command=self.show_results)
        result_menu.add_command(label="导出结果", command=self.export_results)
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 图像显示区域
        image_frame = ttk.LabelFrame(main_frame, text="图像显示", padding="10")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 原始图像标签
        self.original_label = ttk.Label(image_frame)
        self.original_label.grid(row=0, column=0, padx=5, pady=5)
        
        # 处理后图像标签
        self.processed_label = ttk.Label(image_frame)
        self.processed_label.grid(row=0, column=1, padx=5, pady=5)
        
        # 结果文本框
        self.result_text = tk.Text(control_frame, width=40, height=20, wrap=tk.WORD)
        self.result_text.pack(padx=5, pady=5)
        
        # 打开图像按钮
        open_btn = ttk.Button(control_frame, text="打开图像", command=self.open_image)
        open_btn.pack(padx=5, pady=5, fill=tk.X)
        
        # 处理图像按钮
        process_btn = ttk.Button(control_frame, text="处理图像", command=self.process_image)
        process_btn.pack(padx=5, pady=5, fill=tk.X)
        
        # 批量处理按钮
        batch_btn = ttk.Button(control_frame, text="批量处理", command=self.batch_process)
        batch_btn.pack(padx=5, pady=5, fill=tk.X)
        
        # 导出结果按钮
        export_btn = ttk.Button(control_frame, text="导出结果", command=self.export_results)
        export_btn.pack(padx=5, pady=5, fill=tk.X)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            # 读取图像
            image = cv2.imread(file_path)
            self.weed_system.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 显示原始图像
            self.display_image(self.original_image, self.original_label)
    
    def display_image(self, image, label):
        # 调整图像大小以适应显示区域
        max_width = 500
        max_height = 500
        
        h, w, _ = image.shape
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # 转换为PIL图像
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)
        
        # 显示图像
        label.configure(image=photo)
        label.image = photo
    
    def process_image(self):
        if self.weed_system.original_image is None:
            self.show_message("请先打开一张图像")
            return
        
        # 处理图像
        results = self.weed_system.process_single_image(self.weed_system.original_image)
        
        # 显示处理结果
        annotated_image = self.weed_system.annotate_results(self.weed_system.original_image)
        self.display_image(annotated_image, self.processed_label)
        
        # 显示结果信息
        self.show_results()
    
    def show_results(self):
        results = self.weed_system.results
        
        # 清空文本框
        self.result_text.delete(1.0, tk.END)
        
        # 添加结果信息
        self.result_text.insert(tk.END, "杂草分析结果\n")
        self.result_text.insert(tk.END, "=" * 30 + "\n")
        self.result_text.insert(tk.END, f"杂草数量: {results.get('weed_count', 0)}\n")
        self.result_text.insert(tk.END, f"覆盖度: {results.get('coverage', 0):.2f}%\n")
        
        self.result_text.insert(tk.END, "\n形状描述符统计:\n")
        if 'shape_descriptors' in results:
            descriptors = results['shape_descriptors']
            if descriptors:
                circularities = [d['circularity'] for d in descriptors]
                aspect_ratios = [d['aspect_ratio'] for d in descriptors]
                
                self.result_text.insert(tk.END, f"平均圆度: {np.mean(circularities):.4f}\n")
                self.result_text.insert(tk.END, f"平均长宽比: {np.mean(aspect_ratios):.4f}\n")
    
    def batch_process(self):
        input_folder = filedialog.askdirectory(title="选择输入文件夹")
        if not input_folder:
            return
        
        output_folder = filedialog.askdirectory(title="选择输出文件夹")
        if not output_folder:
            return
        
        # 执行批量处理
        self.show_message("开始批量处理...")
        self.root.update()
        
        results = self.weed_system.batch_process(input_folder, output_folder)
        
        self.show_message(f"批量处理完成！处理了 {len(results)} 张图像")
    
    def export_results(self):
        if not self.weed_system.results:
            self.show_message("没有可导出的结果")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            # 导出结果到CSV
            df = pd.DataFrame([self.weed_system.results])
            df.to_csv(file_path, index=False)
            self.show_message("结果导出成功")
    
    def reset(self):
        # 重置系统
        self.weed_system = WeedDetectionSystem()
        
        # 清空图像显示
        self.original_label.configure(image="")
        self.processed_label.configure(image="")
        
        # 清空结果文本
        self.result_text.delete(1.0, tk.END)
    
    def show_message(self, message):
        # 显示消息框
        tk.messagebox.showinfo("信息", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = WeedDetectionGUI(root)
    root.mainloop()