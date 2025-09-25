#!/usr/bin/env python3
"""
POD商品模板和图案预览系统
功能包括：商品模板管理、图案上传预览、图像处理、批量生成等
"""

import tkinter as tk
from tkinter import simpledialog
from tkinter import ttk, filedialog, messagebox, colorchooser
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import sqlite3
import os
import json
from pathlib import Path
import math
from typing import List, Tuple, Dict, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
from batchgengratedialog import BatchGenerateDialog
from imageprocessor import ImageProcessor
from templatemanager import TemplateManager
from datadasemanager import DatabaseManager
from templateeditdialog import TemplateEditDialog



class PODPreviewSystem:
    """POD预览系统主类"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("POD商品模板和图案预览系统")
        # 设置窗口大小
        w, h = 1400, 900
        # 获取屏幕尺寸
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        # 计算居中位置
        x = (sw - w) // 2
        y = (sh - h) // 2
        # 设置大小和位置
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        # 初始化组件
        self.db_manager = DatabaseManager()
        self.template_manager = TemplateManager(self.db_manager)
        self.image_processor = ImageProcessor()
        
        # 状态变量
        self.current_template = None
        self.current_pattern = None
        self.current_preview = None
        self.selected_print_area = None
        self.pattern_effects = {
            'rotation': 0,
            'shadow': False,
            'blur': 0,
            'grayscale': False
        }
        
        # 创建界面
        self.create_interface()
        self.load_templates_list()

        # 添加缩放相关变量
        self.zoom_scale = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
    
    def create_interface(self):
        """创建用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧面板 - 模板和控制
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # 模板选择区域
        self.create_template_section(left_panel)
        # 图案上传区域
        self.create_pattern_section(left_panel)
        # 效果控制区域
        self.create_effects_section(left_panel)
        # 批量处理区域
        self.create_batch_section(left_panel)
        
        # 右侧面板 - 预览区域
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.create_preview_section(right_panel)
    
    def create_template_section(self, parent):
        """创建模板选择区域"""
        template_frame = ttk.LabelFrame(parent, text="模板管理", padding=10)
        template_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 模板列表
        list_frame = ttk.Frame(template_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.template_listbox = tk.Listbox(list_frame, height=8)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.template_listbox.yview)
        self.template_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.template_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.template_listbox.bind('<<ListboxSelect>>', self.on_template_select)
        
        # 模板操作按钮
        btn_frame = ttk.Frame(template_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="上传模板", command=self.upload_template).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="编辑模板", command=self.edit_template).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(btn_frame, text="删除模板", command=self.delete_template).pack(side=tk.RIGHT)
    
    def create_pattern_section(self, parent):
        """创建图案上传区域"""
        pattern_frame = ttk.LabelFrame(parent, text="图案管理", padding=10)
        pattern_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 图案信息
        self.pattern_info = ttk.Label(pattern_frame, text="未选择图案")
        self.pattern_info.pack(fill=tk.X)
        
        # 按钮
        btn_frame = ttk.Frame(pattern_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="上传图案", command=self.upload_pattern).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="清除图案", command=self.clear_pattern).pack(side=tk.RIGHT)
    
    def create_effects_section(self, parent):
        """创建效果控制区域"""
        effects_frame = ttk.LabelFrame(parent, text="图案效果", padding=10)
        effects_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 旋转控制
        rotation_frame = ttk.Frame(effects_frame)
        rotation_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rotation_frame, text="旋转:").pack(side=tk.LEFT)
        
        rotation_buttons_frame = ttk.Frame(rotation_frame)
        rotation_buttons_frame.pack(side=tk.RIGHT)
        ttk.Button(rotation_buttons_frame, text="90°", width=5, 
                  command=lambda: self.rotate_pattern(90)).pack(side=tk.LEFT, padx=1)
        ttk.Button(rotation_buttons_frame, text="180°", width=5,
                  command=lambda: self.rotate_pattern(180)).pack(side=tk.LEFT, padx=1)
        ttk.Button(rotation_buttons_frame, text="270°", width=5,
                  command=lambda: self.rotate_pattern(270)).pack(side=tk.LEFT, padx=1)
        
        # 阴影效果
        shadow_frame = ttk.Frame(effects_frame)
        shadow_frame.pack(fill=tk.X, pady=2)
        ttk.Label(shadow_frame, text="阴影:").pack(side=tk.LEFT)
        self.shadow_var = tk.BooleanVar()
        ttk.Checkbutton(shadow_frame, variable=self.shadow_var, 
                       command=self.update_effects).pack(side=tk.RIGHT)
        
        # 模糊效果
        blur_frame = ttk.Frame(effects_frame)
        blur_frame.pack(fill=tk.X, pady=2)
        ttk.Label(blur_frame, text="模糊:").pack(side=tk.LEFT)
        self.blur_var = tk.IntVar()
        blur_scale = ttk.Scale(blur_frame, from_=0, to=20, variable=self.blur_var,
                              orient=tk.HORIZONTAL, command=self.update_effects)
        blur_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # 黑白效果
        grayscale_frame = ttk.Frame(effects_frame)
        grayscale_frame.pack(fill=tk.X, pady=2)
        ttk.Label(grayscale_frame, text="黑白:").pack(side=tk.LEFT)
        self.grayscale_var = tk.BooleanVar()
        ttk.Checkbutton(grayscale_frame, variable=self.grayscale_var,
                       command=self.update_effects).pack(side=tk.RIGHT)
        
        # 重置按钮
        ttk.Button(effects_frame, text="重置效果", command=self.reset_effects).pack(fill=tk.X, pady=(10, 0))
    
    def create_batch_section(self, parent):
        """创建批量处理区域"""
        batch_frame = ttk.LabelFrame(parent, text="批量处理", padding=10)
        batch_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(batch_frame, text="批量上传图案", command=self.batch_upload_patterns).pack(fill=tk.X, pady=2)
        ttk.Button(batch_frame, text="批量生成预览", command=self.batch_generate).pack(fill=tk.X, pady=2)
        
        # 导出选项
        export_frame = ttk.Frame(batch_frame)
        export_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(export_frame, text="导出倍数:").pack(side=tk.LEFT)
        self.export_scale_var = tk.StringVar(value="2x")
        scale_combo = ttk.Combobox(export_frame, textvariable=self.export_scale_var,
                                  values=["1x", "2x", "3x", "4x", "自定义"], width=8)
        scale_combo.pack(side=tk.RIGHT)
        
        ttk.Button(batch_frame, text="导出预览", command=self.export_preview).pack(fill=tk.X, pady=(5, 0))
    
    def create_preview_section(self, parent):
        """创建预览区域"""
        preview_frame = ttk.LabelFrame(parent, text="预览区域", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        # 创建工具栏
        toolbar = ttk.Frame(preview_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(toolbar, text="适应画布", command=self.fit_to_canvas).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="实际大小", command=self.actual_size).pack(side=tk.LEFT, padx=2)
        ttk.Label(toolbar, text="缩放:").pack(side=tk.LEFT, padx=(10, 2))

        # 创建画布
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas = tk.Canvas(canvas_frame, bg='white')
        
        # 滚动条
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.preview_canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        
        self.preview_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # 布局
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # 绑定事件
        self.preview_canvas.bind("<Button-1>", self.on_canvas_click)
        self.preview_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.preview_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.preview_canvas.bind("<Configure>", self.on_canvas_resize)
        
        # 状态栏
        status_frame = ttk.Frame(preview_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="准备就绪")
        self.status_label.pack(side=tk.LEFT)
        
        self.progress = ttk.Progressbar(status_frame, length=200)
        self.progress.pack(side=tk.RIGHT)
    
    def on_canvas_click(self, event):
        """画布点击事件"""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def on_canvas_resize(self, event):
        """画布大小改变事件"""
        if self.current_preview is not None:
            self.display_preview(self.current_preview)
    

    def on_canvas_drag(self, event):
        """画布拖拽事件"""
        if not self.is_panning:
            return
        
        # 计算移动距离
        dx = (self.pan_start_x - event.x) / self.zoom_scale
        dy = (self.pan_start_y - event.y) / self.zoom_scale
        
        # 移动画布
        self.preview_canvas.scan_dragto(event.x, event.y, gain=1)
        self.preview_canvas.scan_mark(self.pan_start_x, self.pan_start_y)
        
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def on_canvas_release(self, event):
        """画布释放事件"""
        self.is_panning = False
    
    def on_mousewheel(self, event):
        """鼠标滚轮事件"""
        if not self.current_preview is not None:
            return
            
        # 获取当前鼠标位置
        x = self.preview_canvas.canvasx(event.x)
        y = self.preview_canvas.canvasy(event.y)
        
        # 根据滚轮方向调整缩放
        if event.delta > 0:
            self.zoom_scale *= 1.1
        else:
            self.zoom_scale /= 1.1
        
        # 限制缩放范围
        self.zoom_scale = max(0.1, min(5.0, self.zoom_scale))
        
        # 重新显示图像
        self.display_preview(self.current_preview)
    
    def load_templates_list(self):
        """加载模板列表"""
        self.template_listbox.delete(0, tk.END)
        
        categories = self.template_manager.get_templates_by_category()
        for category, templates in categories.items():
            self.template_listbox.insert(tk.END, f"--- {category} ---")
            for template in templates:
                self.template_listbox.insert(tk.END, f"  {template['name']}")

    def fit_to_canvas(self):
        """适应画布大小"""
        if self.current_preview is None:
            return
            
        self.zoom_scale = 1.0
        self.display_preview(self.current_preview)
    def actual_size(self):
        """实际大小显示"""
        if self.current_preview is None:
            return
            
        self.zoom_scale = 1.0
        self.display_preview(self.current_preview)

    def on_template_select(self, event):
        """模板选择事件"""
        selection = self.template_listbox.curselection()
        if not selection:
            return
        
        selected_text = self.template_listbox.get(selection[0])
        if selected_text.startswith("---"):
            return
        
        template_name = selected_text.strip()
        
        # 查找模板
        for template in self.template_manager.templates:
            if template['name'] == template_name:
                self.current_template = template
                self.load_template_preview()
                break
    
    def load_template_preview(self):
        """加载模板预览"""
        if not self.current_template:
            return
        
        try:
            # 加载模板图像
            img_path = self.current_template['image_path']
            if os.path.exists(img_path):
                template_img = cv2.imread(img_path)
            else:
                # 如果文件不存在，创建默认模板
                template_img = self.create_default_template_image()
            
            self.display_preview(template_img)
            self.status_label.config(text=f"已加载模板: {self.current_template['name']}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载模板失败: {e}")
    
    def create_default_template_image(self) -> np.ndarray:
        """创建默认模板图像"""
        img = np.ones((400, 300, 3), dtype=np.uint8) * 240
        
        # 绘制打印区域
        for area in self.current_template['print_areas']:
            cv2.rectangle(img,
                         (area['x'], area['y']),
                         (area['x'] + area['width'], area['y'] + area['height']),
                         (200, 200, 200), 2)
        return img
    
    def upload_template(self):
        """上传自定义模板"""
        file_path = filedialog.askopenfilename(
            title="选择模板图像",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if not file_path:
            return
        
        try:
            # 获取file_path 文件名称
            template_name = os.path.basename(file_path)
            img_path = f"templates/{template_name.replace(' ', '_')}.png"

            # 使用numpy读取图片，避免中文路径问题
            with open(file_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

            cv2.imwrite(img_path, img)

            if img is None:
                messagebox.showerror("错误", "无法读取所选图像文件")
                return
                
            # 弹出对话框设置模板信息
            dialog = TemplateEditDialog(self.root, img_path)
            if dialog.result:
                template_data = dialog.result
                # 保存模板
                if self.template_manager.add_custom_template(
                    template_data['name'],
                    template_data['category'],
                    img_path,
                    template_data['print_areas']
                ):
                    self.template_manager.load_templates()
                    self.load_templates_list()
                    messagebox.showinfo("成功", "模板上传成功!")
                else:
                    messagebox.showerror("错误", "模板保存失败!")
                    
        except Exception as e:
            messagebox.showerror("错误", f"上传模板失败: {str(e)}")
    
    def upload_pattern(self):
        """上传图案"""
        file_path = filedialog.askopenfilename(
            title="选择图案文件",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )
        
        if not file_path:
            return
        
        # 使用numpy读取图片，避免中文路径问题
        with open(file_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), np.uint8)
            self.current_pattern = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        
        # 确保图像有正确的通道数
        if self.current_pattern is not None:
            if len(self.current_pattern.shape) == 2:  # 灰度图
                self.current_pattern = cv2.cvtColor(self.current_pattern, cv2.COLOR_GRAY2BGRA)
            elif self.current_pattern.shape[2] == 3:  # BGR图
                self.current_pattern = cv2.cvtColor(self.current_pattern, cv2.COLOR_BGR2BGRA)
            elif self.current_pattern.shape[2] == 4:  # BGRA图
                pass  # 已经是4通道，无需转换
        
        # 更新界面
        filename = os.path.basename(file_path)
        self.pattern_info.config(text=f"已加载: {filename}")
        # 重置效果
        self.reset_effects()
        # 生成预览
        self.generate_preview()
            
    def clear_pattern(self):
        """清除图案"""
        self.current_pattern = None
        self.pattern_info.config(text="未选择图案")
        self.reset_effects()
        if self.current_template:
            self.load_template_preview()
    
    def rotate_pattern(self, angle: int):
        """旋转图案"""
        if self.current_pattern is None:
            return
        
        if angle == 90:
            self.current_pattern = cv2.rotate(self.current_pattern, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            self.current_pattern = cv2.rotate(self.current_pattern, cv2.ROTATE_180)
        elif angle == 270:
            self.current_pattern = cv2.rotate(self.current_pattern, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        self.pattern_effects['rotation'] = (self.pattern_effects['rotation'] + angle) % 360
        self.generate_preview()
    
    def update_effects(self, *args):
        """更新效果设置"""
        if self.current_pattern is None:
            return
        
        self.pattern_effects['shadow'] = self.shadow_var.get()
        self.pattern_effects['blur'] = self.blur_var.get()
        self.pattern_effects['grayscale'] = self.grayscale_var.get()
        
        self.generate_preview()
    
    def reset_effects(self):
        """重置所有效果"""
        self.pattern_effects = {
            'rotation': 0,
            'shadow': False,
            'blur': 0,
            'grayscale': False
        }
        
        # 更新界面
        self.shadow_var.set(False)
        self.blur_var.set(0)
        self.grayscale_var.set(False)
        
        if self.current_pattern is not None:
            self.generate_preview()
    
    def generate_preview(self):
        """生成预览图"""
        if not self.current_template or self.current_pattern is None:
            return
        
        # 加载模板图像
        if os.path.exists(self.current_template['image_path']):
            template_img = cv2.imread(self.current_template['image_path'])
        else:
            template_img = self.create_default_template_image()
        
        # 复制模板图像
        result_img = template_img.copy()
        
        # 处理每个打印区域
        for area in self.current_template['print_areas']:
            print(area)
            processed_pattern = self.apply_pattern_effects()
            
            # 调整图案大小适应打印区域
            area_size = (area['width'], area['height'])
            resized_pattern = self.image_processor.resize_with_aspect_ratio(
                processed_pattern, area_size, keep_aspect=True)
            # 应用透视变换
            src_points = np.array([
                [0, 0],
                [resized_pattern.shape[1], 0],
                [resized_pattern.shape[1], resized_pattern.shape[0]],
                [0, resized_pattern.shape[0]]
            ], dtype=np.float32)
            
            dst_points = np.array(area['points'], dtype=np.float32)
            resized_pattern = self.image_processor.perspective_transform(
                resized_pattern, src_points, dst_points)
            
            # 合成到模板上
            result_img = self.composite_pattern_on_template(
                result_img, resized_pattern, area)
        
        self.current_preview = result_img
        self.display_preview(result_img)
            
    def apply_pattern_effects(self) -> np.ndarray:
        """应用图案效果"""
        if self.current_pattern is None:
            return np.array([])
        
        pattern = self.current_pattern.copy()
        
        # 应用灰度效果
        if self.pattern_effects['grayscale']:
            pattern = self.image_processor.to_grayscale(pattern, preserve_alpha=True)
        
        # 应用模糊效果
        if self.pattern_effects['blur'] > 0:
            pattern = self.image_processor.apply_blur(pattern, self.pattern_effects['blur'])
        
        # 应用阴影效果
        if self.pattern_effects['shadow']:
            pattern = self.image_processor.add_shadow(pattern)
        
        return pattern
    
    def composite_pattern_on_template(self, template: np.ndarray, pattern: np.ndarray, 
                                    area: Dict) -> np.ndarray:
        """将图案合成到模板上"""
        if pattern.size == 0:
            return template
        
        # 确保图案不超出打印区域
        area_h, area_w = area['height'], area['width']
        pattern_h, pattern_w = pattern.shape[:2]
        
        # 计算放置位置（居中）
        start_x = area['x'] + (area_w - pattern_w) // 2
        start_y = area['y'] + (area_h - pattern_h) // 2
        
        # 确保坐标在有效范围内
        start_x = max(0, min(start_x, template.shape[1] - pattern_w))
        start_y = max(0, min(start_y, template.shape[0] - pattern_h))
        
        # 计算实际可放置的区域
        end_x = min(start_x + pattern_w, template.shape[1])
        end_y = min(start_y + pattern_h, template.shape[0])
        
        actual_w = end_x - start_x
        actual_h = end_y - start_y
        
        if actual_w <= 0 or actual_h <= 0:
            return template
        
        # 获取要放置的图案区域
        pattern_roi = pattern[:actual_h, :actual_w]
        
        # 如果图案有alpha通道，进行alpha合成
        if len(pattern_roi.shape) == 3 and pattern_roi.shape[2] == 4:
            # Alpha合成
            template_roi = template[start_y:end_y, start_x:end_x]
            
            # 确保模板区域有alpha通道
            if template_roi.shape[2] == 3:
                alpha = np.ones((template_roi.shape[0], template_roi.shape[1], 1), 
                              dtype=template_roi.dtype) * 255
                template_roi = np.concatenate([template_roi, alpha], axis=2)
            
            # 执行alpha合成
            blended = self.image_processor.blend_alpha(template_roi, pattern_roi)
            
            # 如果原模板只有3个通道，去掉alpha通道
            if template.shape[2] == 3:
                blended = blended[:, :, :3]
            
            template[start_y:end_y, start_x:end_x] = blended
        else:
            # 直接替换
            if len(pattern_roi.shape) == 3 and pattern_roi.shape[2] == 3:
                template[start_y:end_y, start_x:end_x] = pattern_roi
            else:
                # 灰度图转换为彩色
                if len(pattern_roi.shape) == 2:
                    pattern_roi = cv2.cvtColor(pattern_roi, cv2.COLOR_GRAY2BGR)
                template[start_y:end_y, start_x:end_x] = pattern_roi
        
        return template
    
    def display_preview(self, img: np.ndarray):
        """显示预览图像"""
        if img is None or img.size == 0:
            return
        
        # 转换为RGB格式
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # 获取画布尺寸
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # 计算适应画布的缩放比例
            img_h, img_w = img_rgb.shape[:2]
            fit_scale = min(canvas_width / img_w, canvas_height / img_h)
            
            # 应用用户缩放
            scale = fit_scale * self.zoom_scale
            
            # 调整图像大小
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为PIL图像并显示
        pil_img = Image.fromarray(img_rgb)
        self.preview_photo = ImageTk.PhotoImage(pil_img)
        
        # 清除画布并显示新图像
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.preview_photo,
            anchor="center"
        )
        
        # 更新滚动区域
        self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all"))
    
    
    def edit_template(self):
        """编辑模板"""
        if not self.current_template:
            messagebox.showwarning("警告", "请先选择一个模板")
            return
        
        dialog = TemplateEditDialog(self.root, self.current_template['image_path'], 
                                  self.current_template)
        if dialog.result:
            # 更新模板数据
            template_data = dialog.result
            if self.db_manager.save_template(
                self.current_template['name'],
                template_data['category'],
                self.current_template['image_path'],
                template_data['print_areas']
            ):
                self.template_manager.load_templates()
                messagebox.showinfo("成功", "模板更新成功!")
            else:
                messagebox.showerror("错误", "模板更新失败!")
    
    def delete_template(self):
        """删除模板"""
        if not self.current_template:
            messagebox.showwarning("警告", "请先选择一个模板")
            return
        
        if messagebox.askyesno("确认", f"确定要删除模板 '{self.current_template['name']}' 吗？"):
            self.db_manager.delete_template(self.current_template['id'])
            self.template_manager.load_templates()
            self.load_templates_list()
            self.current_template = None
            self.preview_canvas.delete("all")
            messagebox.showinfo("成功", "模板删除成功!")
    
    def batch_upload_patterns(self):
        """批量上传图案"""
        file_paths = filedialog.askopenfilenames(
            title="选择多个图案文件",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )
        
        if not file_paths:
            return
        
        self.batch_patterns = []
        for file_path in file_paths:
            try:
                pattern = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if pattern is not None:
                    self.batch_patterns.append({
                        'name': os.path.basename(file_path),
                        'image': pattern
                    })
            except Exception as e:
                print(f"加载 {file_path} 失败: {e}")
        
        messagebox.showinfo("成功", f"成功加载 {len(self.batch_patterns)} 个图案")
    
    def batch_generate(self):
        """批量生成预览"""
        if not hasattr(self, 'batch_patterns') or not self.batch_patterns:
            messagebox.showwarning("警告", "请先批量上传图案")
            return
        
        if not self.template_manager.templates:
            messagebox.showwarning("警告", "没有可用的模板")
            return
        
        # 选择输出目录
        output_dir = filedialog.askdirectory(title="选择输出目录")
        if not output_dir:
            return
        
        # 创建批量生成对话框
        dialog = BatchGenerateDialog(self.root, self.template_manager.templates, 
                                   self.batch_patterns)
        
        if dialog.result:
            selected_templates = dialog.result['templates']
            selected_patterns = dialog.result['patterns']
            
            self.run_batch_generation(selected_templates, selected_patterns, output_dir)
    
    def run_batch_generation(self, templates: List[Dict], patterns: List[Dict], 
                           output_dir: str):
        """执行批量生成"""
        total_tasks = len(templates) * len(patterns)
        current_task = 0
        
        self.progress.config(maximum=total_tasks, value=0)
        
        def generate_task():
            nonlocal current_task
            
            for template in templates:
                for pattern in patterns:
                    try:
                        # 临时设置当前模板和图案
                        old_template = self.current_template
                        old_pattern = self.current_pattern
                        
                        self.current_template = template
                        self.current_pattern = pattern['image']
                        
                        # 生成预览
                        self.generate_preview()
                        
                        if self.current_preview is not None:
                            # 保存预览
                            filename = f"{template['name']}_{pattern['name']}"
                            output_path = os.path.join(output_dir, f"{filename}.png")
                            cv2.imwrite(output_path, self.current_preview)
                        
                        # 恢复原设置
                        self.current_template = old_template
                        self.current_pattern = old_pattern
                        
                        current_task += 1
                        self.progress.config(value=current_task)
                        self.status_label.config(text=f"生成中... {current_task}/{total_tasks}")
                        self.root.update_idletasks()
                        
                    except Exception as e:
                        print(f"生成 {template['name']} x {pattern['name']} 失败: {e}")
            
            self.progress.config(value=0)
            self.status_label.config(text="批量生成完成")
            messagebox.showinfo("成功", f"批量生成完成！\n共生成 {current_task} 个预览图")
        
        # 在新线程中运行生成任务
        threading.Thread(target=generate_task, daemon=True).start()
    
    def export_preview(self):
        """导出预览"""
        if self.current_preview is None:
            messagebox.showwarning("警告", "没有可导出的预览图")
            return
        
        # 获取导出倍数
        scale_text = self.export_scale_var.get()
        if scale_text == "自定义":
            scale_dialog = tk.simpledialog.askfloat("自定义倍数", "请输入导出倍数:", 
                                                   minvalue=0.1, maxvalue=10.0)
            if scale_dialog is None:
                return
            scale_factor = scale_dialog
        else:
            scale_factor = float(scale_text.replace('x', ''))
        
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="保存预览图",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg")]
        )
        
        if not file_path:
            return
        
        try:
            # 缩放图像
            if scale_factor != 1.0:
                h, w = self.current_preview.shape[:2]
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                scaled_img = cv2.resize(self.current_preview, (new_w, new_h), 
                                      interpolation=cv2.INTER_LANCZOS4)
            else:
                scaled_img = self.current_preview
            
            # 保存图像
            cv2.imwrite(file_path, scaled_img)
            messagebox.showinfo("成功", f"预览图已保存到: {file_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")
    
    def run(self):
        """运行应用程序"""
        self.root.mainloop()

def main():
    """主函数"""
    try:
        # 创建必要的目录
        os.makedirs("templates", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        
        # 启动应用程序
        app = PODPreviewSystem()
        app.run()
        
    except Exception as e:
        print(f"程序启动失败: {e}")
        messagebox.showerror("错误", f"程序启动失败: {e}")


if __name__ == "__main__":
    main()