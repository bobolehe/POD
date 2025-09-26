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

class TemplateEditDialog:
    """模板编辑对话框"""
    
    def __init__(self, parent, image_path: str, template_data: Dict = None):
        self.result = None
        self.image_path = image_path
        self.template_data = template_data or {}
        
        # 创建对话框窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("编辑模板")
        # 设置窗口大小
        w, h = 1440, 950
        # 获取屏幕尺寸
        sw = self.dialog.winfo_screenwidth()
        sh = self.dialog.winfo_screenheight()
        # 计算居中位置
        x = (sw - w) // 2
        y = (sh - h) // 2
        # 设置大小和位置
        print(f"{w}x{h}+{x}+{y}")
        self.dialog.geometry(f"{w}x{h}+{x}+{y}")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # 初始化变量
        self.print_areas = self.template_data.get('print_areas', [])
        self.mirror_areas = self.template_data.get('mirror_areas', [])  # 新增：镜像补充区域
        self.current_area = None
        self.current_mirror_area = None  # 新增：当前选中的镜像区域
        self.template_img = None
        self.scale_factor = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.picking_point = None
        self.original_cursor = None
        self.point_picking_sequence = None
        self.area_type = "print"  # 新增：当前编辑的区域类型，"print" 或 "mirror"
        
        # 拖动相关变量
        self.drag_start = None
        self.drag_rect = None
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.last_canvas_x = 0
        self.last_canvas_y = 0
        
        self.create_interface()
        self.load_image()
        self.dialog.update_idletasks()
        self.dialog.after(100, self.fit_to_canvas)

        # 等待对话框关闭
        self.dialog.wait_window()
    
    def create_interface(self):
        """创建对话框界面"""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.Frame(main_frame, width=330)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 模板信息
        info_frame = ttk.LabelFrame(control_frame, text="模板信息", padding=5)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(info_frame, text="名称:").grid(row=0, column=0, sticky="w", pady=2)
        self.name_var = tk.StringVar(value=self.template_data.get('name', ''))
        ttk.Entry(info_frame, textvariable=self.name_var).grid(row=0, column=1, sticky="ew", pady=2)
        
        ttk.Label(info_frame, text="分类:").grid(row=1, column=0, sticky="w", pady=2)
        self.category_var = tk.StringVar(value=self.template_data.get('category', ''))
        category_combo = ttk.Combobox(info_frame, textvariable=self.category_var,
                                    values=["T恤", "杯子", "家居", "印刷品", "其他"])
        category_combo.grid(row=1, column=1, sticky="ew", pady=2)
        
        info_frame.columnconfigure(1, weight=1)
        
        # 打印区域管理
        area_frame = ttk.LabelFrame(control_frame, text="打印区域", padding=5)
        area_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # 区域列表
        self.area_listbox = tk.Listbox(area_frame, height=3)
        self.area_listbox.pack(fill=tk.X, pady=(0, 10))
        self.area_listbox.bind('<<ListboxSelect>>', self.on_area_select)
        
        # 区域操作按钮
        btn_frame = ttk.Frame(area_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="添加区域", command=self.add_area).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="删除区域", command=self.delete_area).pack(side=tk.RIGHT)
        
        # 区域属性 - 改为四个坐标点
        prop_frame = ttk.LabelFrame(area_frame, text="区域坐标 (四个角点)", padding=5)
        prop_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 左上角坐标
        ttk.Label(prop_frame, text="左上角:").grid(row=0, column=0, sticky="w", columnspan=2)
        self.tl_x_var = tk.IntVar()
        self.tl_y_var = tk.IntVar()
        ttk.Label(prop_frame, text="X:").grid(row=1, column=0, sticky="w")
        ttk.Entry(prop_frame, textvariable=self.tl_x_var, width=8).grid(row=1, column=1, padx=(5, 0))
        ttk.Label(prop_frame, text="Y:").grid(row=1, column=2, sticky="w")
        ttk.Entry(prop_frame, textvariable=self.tl_y_var, width=8).grid(row=1, column=3, padx=(5, 0))
        ttk.Button(prop_frame, text="拾取", width=4, command=lambda: self.start_pick_point("tl")).grid(row=1, column=4, padx=(5, 0))
        
        # 右上角坐标
        ttk.Label(prop_frame, text="右上角:").grid(row=2, column=0, sticky="w", columnspan=2)
        self.tr_x_var = tk.IntVar()
        self.tr_y_var = tk.IntVar()
        ttk.Label(prop_frame, text="X:").grid(row=3, column=0, sticky="w")
        ttk.Entry(prop_frame, textvariable=self.tr_x_var, width=8).grid(row=3, column=1, padx=(5, 0))
        ttk.Label(prop_frame, text="Y:").grid(row=3, column=2, sticky="w")
        ttk.Entry(prop_frame, textvariable=self.tr_y_var, width=8).grid(row=3, column=3, padx=(5, 0))
        ttk.Button(prop_frame, text="拾取", width=4, command=lambda: self.start_pick_point("tr")).grid(row=3, column=4, padx=(5, 0))
        
        # 右下角坐标
        ttk.Label(prop_frame, text="右下角:").grid(row=4, column=0, sticky="w", columnspan=2)
        self.br_x_var = tk.IntVar()
        self.br_y_var = tk.IntVar()
        ttk.Label(prop_frame, text="X:").grid(row=5, column=0, sticky="w")
        ttk.Entry(prop_frame, textvariable=self.br_x_var, width=8).grid(row=5, column=1, padx=(5, 0))
        ttk.Label(prop_frame, text="Y:").grid(row=5, column=2, sticky="w")
        ttk.Entry(prop_frame, textvariable=self.br_y_var, width=8).grid(row=5, column=3, padx=(5, 0))
        ttk.Button(prop_frame, text="拾取", width=4, command=lambda: self.start_pick_point("br")).grid(row=5, column=4, padx=(5, 0))
        
        # 左下角坐标
        ttk.Label(prop_frame, text="左下角:").grid(row=6, column=0, sticky="w", columnspan=2)
        self.bl_x_var = tk.IntVar()
        self.bl_y_var = tk.IntVar()
        ttk.Label(prop_frame, text="X:").grid(row=7, column=0, sticky="w")
        ttk.Entry(prop_frame, textvariable=self.bl_x_var, width=8).grid(row=7, column=1, padx=(5, 0))
        ttk.Label(prop_frame, text="Y:").grid(row=7, column=2, sticky="w")
        ttk.Entry(prop_frame, textvariable=self.bl_y_var, width=8).grid(row=7, column=3, padx=(5, 0))
        ttk.Button(prop_frame, text="拾取", width=4, command=lambda: self.start_pick_point("bl")).grid(row=7, column=4, padx=(5, 0))
        
        ttk.Button(prop_frame, text="更新区域", command=self.update_area).grid(row=9, column=0, columnspan=2, pady=(5, 0))
        ttk.Button(prop_frame, text="矩形模式", command=self.set_rectangle_mode).grid(row=9, column=2, columnspan=2, pady=(5, 0))
        ttk.Button(prop_frame, text="自由模式", command=self.set_free_mode).grid(row=9, column=4,  columnspan=2, pady=(5, 0))
                                                                                
        # 补充区域管理（镜像填充）
        mirror_frame = ttk.LabelFrame(control_frame, text="补充区域（镜像填充）", padding=10)
        mirror_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 镜像区域列表
        self.mirror_listbox = tk.Listbox(mirror_frame, height=3)
        self.mirror_listbox.pack(fill=tk.X, pady=(0, 10))
        self.mirror_listbox.bind('<<ListboxSelect>>', self.on_mirror_area_select)
        
        # 镜像区域操作按钮
        mirror_btn_frame = ttk.Frame(mirror_frame)
        mirror_btn_frame.pack(fill=tk.X)
        
        ttk.Button(mirror_btn_frame, text="添加补充区域", command=self.add_mirror_area).pack(side=tk.LEFT)
        ttk.Button(mirror_btn_frame, text="删除补充区域", command=self.delete_mirror_area).pack(side=tk.RIGHT)
        
        # 镜像区域属性
        mirror_prop_frame = ttk.LabelFrame(mirror_frame, text="补充区域坐标", padding=5)
        mirror_prop_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 左上角坐标
        ttk.Label(mirror_prop_frame, text="左上角:").grid(row=0, column=0, sticky="w", columnspan=2)
        ttk.Label(mirror_prop_frame, text="X:").grid(row=1, column=0, sticky="w")
        self.mirror_tl_x_var = tk.IntVar()
        ttk.Entry(mirror_prop_frame, textvariable=self.mirror_tl_x_var, width=8).grid(row=1, column=1, padx=(5, 10))
        ttk.Label(mirror_prop_frame, text="Y:").grid(row=1, column=2, sticky="w")
        self.mirror_tl_y_var = tk.IntVar()
        ttk.Entry(mirror_prop_frame, textvariable=self.mirror_tl_y_var, width=8).grid(row=1, column=3, padx=(5, 0))
        ttk.Button(mirror_prop_frame, text="拾取", width=4, 
                    command=lambda: self.start_pick_mirror_point("tl")).grid(row=1, column=4, padx=(5, 0))
        
        # 右上角坐标
        ttk.Label(mirror_prop_frame, text="右上角:").grid(row=2, column=0, sticky="w", columnspan=2)
        ttk.Label(mirror_prop_frame, text="X:").grid(row=3, column=0, sticky="w")
        self.mirror_tr_x_var = tk.IntVar()
        ttk.Entry(mirror_prop_frame, textvariable=self.mirror_tr_x_var, width=8).grid(row=3, column=1, padx=(5, 10))
        ttk.Label(mirror_prop_frame, text="Y:").grid(row=3, column=2, sticky="w")
        self.mirror_tr_y_var = tk.IntVar()
        ttk.Entry(mirror_prop_frame, textvariable=self.mirror_tr_y_var, width=8).grid(row=3, column=3, padx=(5, 0))
        ttk.Button(mirror_prop_frame, text="拾取", width=4,
                    command=lambda: self.start_pick_mirror_point("tr")).grid(row=3, column=4, padx=(5, 0))
        
        # 右下角坐标
        ttk.Label(mirror_prop_frame, text="右下角:").grid(row=4, column=0, sticky="w", columnspan=2)
        ttk.Label(mirror_prop_frame, text="X:").grid(row=5, column=0, sticky="w")
        self.mirror_br_x_var = tk.IntVar()
        ttk.Entry(mirror_prop_frame, textvariable=self.mirror_br_x_var, width=8).grid(row=5, column=1, padx=(5, 10))
        ttk.Label(mirror_prop_frame, text="Y:").grid(row=5, column=2, sticky="w")
        self.mirror_br_y_var = tk.IntVar()
        ttk.Entry(mirror_prop_frame, textvariable=self.mirror_br_y_var, width=8).grid(row=5, column=3, padx=(5, 0))
        ttk.Button(mirror_prop_frame, text="拾取", width=4,
                    command=lambda: self.start_pick_mirror_point("br")).grid(row=5, column=4, padx=(5, 0))
        
        # 左下角坐标
        ttk.Label(mirror_prop_frame, text="左下角:").grid(row=6, column=0, sticky="w", columnspan=2)
        ttk.Label(mirror_prop_frame, text="X:").grid(row=7, column=0, sticky="w")
        self.mirror_bl_x_var = tk.IntVar()
        ttk.Entry(mirror_prop_frame, textvariable=self.mirror_bl_x_var, width=8).grid(row=7, column=1, padx=(5, 10))
        ttk.Label(mirror_prop_frame, text="Y:").grid(row=7, column=2, sticky="w")
        self.mirror_bl_y_var = tk.IntVar()
        ttk.Entry(mirror_prop_frame, textvariable=self.mirror_bl_y_var, width=8).grid(row=7, column=3, padx=(5, 0))
        ttk.Button(mirror_prop_frame, text="拾取", width=4,
                    command=lambda: self.start_pick_mirror_point("bl")).grid(row=7, column=4, padx=(5, 0))
        
        ttk.Button(mirror_prop_frame, text="更新补充区域", command=self.update_mirror_area).grid(row=8, column=0, 
                                                                                                    columnspan=4, pady=(10, 0))
        
        # 按钮区域
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="确定", command=self.confirm).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="取消", command=self.cancel).pack(side=tk.RIGHT)
        
        # 右侧预览区域
        preview_frame = ttk.LabelFrame(main_frame, text="模板预览", padding=10)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 工具栏
        toolbar_frame = ttk.Frame(preview_frame)
        toolbar_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar_frame, text="放大", command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(toolbar_frame, text="缩小", command=self.zoom_out).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(toolbar_frame, text="适应", command=self.fit_to_canvas).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(toolbar_frame, text="1:1", command=self.reset_zoom).pack(side=tk.LEFT, padx=(5, 0))
        
        self.zoom_label = ttk.Label(toolbar_frame, text="缩放: 100%")
        self.zoom_label.pack(side=tk.LEFT, padx=(20, 0))
        
        self.mode_label = ttk.Label(toolbar_frame, text="模式: 矩形", foreground="blue")
        self.mode_label.pack(side=tk.RIGHT)
        
        # 画布框架
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建画布和滚动条
        self.canvas = tk.Canvas(canvas_frame, bg='lightgray')
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # 布局滚动条和画布
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # 绑定画布事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_right_click)  # 右键平移
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # 鼠标滚轮缩放
        self.canvas.bind("<Control-Button-4>", self.on_mouse_wheel)  # Linux
        self.canvas.bind("<Control-Button-5>", self.on_mouse_wheel)  # Linux
        
        # 编辑模式
        self.edit_mode = "rectangle"  # "rectangle" 或 "free"
    
    def load_image(self):
        """加载模板图像"""
        try:
            self.template_img = cv2.imread(self.image_path)
            if self.template_img is not None:
                self.img_height, self.img_width = self.template_img.shape[:2]
                self.fit_to_canvas()
                self.update_area_list()
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {e}")
    
    def start_pick_point(self, point_type: str):
        """开始拾取坐标点"""
        self.picking_point = point_type
        self.original_cursor = self.canvas.cget("cursor")
        self.canvas.config(cursor="crosshair")

    def end_pick_point(self):
        """结束拾取坐标点"""
        self.picking_point = None
        self.canvas.config(cursor=self.original_cursor)

    def zoom_in(self):
        """放大"""
        self.scale_factor *= 1.2
        self.update_zoom_label()
        self.display_template()
    
    def zoom_out(self):
        """缩小"""
        self.scale_factor /= 1.2
        self.update_zoom_label()
        self.display_template()
    
    def reset_zoom(self):
        """重置缩放为1:1"""
        self.scale_factor = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.update_zoom_label()
        self.display_template()
    
    def fit_to_canvas(self):
        """适应画布大小"""
        if self.template_img is None:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # 计算合适的缩放比例
            scale_x = (canvas_width - 50) / self.img_width
            scale_y = (canvas_height - 50) / self.img_height
            self.scale_factor = min(scale_x, scale_y, 1.0)  # 最大不超过原始大小
            
            # 居中显示
            scaled_width = self.img_width * self.scale_factor
            scaled_height = self.img_height * self.scale_factor
            self.canvas_offset_x = (canvas_width - scaled_width) / 2
            self.canvas_offset_y = (canvas_height - scaled_height) / 2
            
            self.update_zoom_label()
            self.display_template()
    
    def update_zoom_label(self):
        """更新缩放标签"""
        zoom_percent = int(self.scale_factor * 100)
        self.zoom_label.config(text=f"缩放: {zoom_percent}%")
    
    def set_rectangle_mode(self):
        """设置矩形编辑模式"""
        self.edit_mode = "rectangle"
        self.mode_label.config(text="模式: 矩形")
    
    def set_free_mode(self):
        """设置自由编辑模式"""
        self.edit_mode = "free"
        self.mode_label.config(text="模式: 自由")
        # 初始化点位拾取序号
        self.point_picking_sequence = 0
        messagebox.showinfo("提示", "请依次点击设置左上、右上、右下、左下四个角点")
    
    def display_template(self):
        """显示模板图像"""
        if self.template_img is None:
            return
        
        # 转换为RGB
        img_rgb = cv2.cvtColor(self.template_img, cv2.COLOR_BGR2RGB)
        
        # 应用缩放
        if self.scale_factor != 1.0:
            new_width = int(self.img_width * self.scale_factor)
            new_height = int(self.img_height * self.scale_factor)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 绘制打印区域
        display_img = img_rgb.copy()
        for i, area in enumerate(self.print_areas):
            color = (255, 0, 0) if i == self.current_area else (0, 255, 0)
            
            if 'points' in area:
                # 四个坐标点模式
                points = np.array(area['points'], dtype=np.int32)
                # 应用缩放和偏移
                scaled_points = []
                for point in points:
                    scaled_x = int(point[0] * self.scale_factor)
                    scaled_y = int(point[1] * self.scale_factor)
                    scaled_points.append([scaled_x, scaled_y])
                
                scaled_points = np.array(scaled_points, dtype=np.int32)
                cv2.polylines(display_img, [scaled_points], True, color, 2)
                
                # 绘制角点
                for j, point in enumerate(scaled_points):
                    cv2.circle(display_img, tuple(point), 4, color, -1)
                    cv2.putText(display_img, str(j+1), (point[0]+5, point[1]-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                # 兼容旧的矩形模式
                x, y, w, h = area['x'], area['y'], area['width'], area['height']
                scaled_x = int(x * self.scale_factor)
                scaled_y = int(y * self.scale_factor)
                scaled_w = int(w * self.scale_factor)
                scaled_h = int(h * self.scale_factor)
                
                cv2.rectangle(display_img, (scaled_x, scaled_y), 
                             (scaled_x + scaled_w, scaled_y + scaled_h), color, 2)
            
            # 添加区域标签
            label_x = int((area.get('x', 0) if 'points' not in area else area['points'][0][0]) * self.scale_factor)
            label_y = int((area.get('y', 0) if 'points' not in area else area['points'][0][1]) * self.scale_factor)
            cv2.putText(display_img, f"Area {i+1}", (label_x + 5, label_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制镜像区域
        for i, area in enumerate(self.mirror_areas):
            color = (255, 255, 0) if i == self.current_mirror_area else (0, 255, 255)  # 黄色/青色
            
            if 'points' in area:
                # 四个坐标点模式
                points = np.array(area['points'], dtype=np.int32)
                # 应用缩放和偏移
                scaled_points = []
                for point in points:
                    scaled_x = int(point[0] * self.scale_factor)
                    scaled_y = int(point[1] * self.scale_factor)
                    scaled_points.append([scaled_x, scaled_y])
                
                scaled_points = np.array(scaled_points, dtype=np.int32)
                cv2.polylines(display_img, [scaled_points], True, color, 2)
                
                # 绘制角点
                for j, point in enumerate(scaled_points):
                    cv2.circle(display_img, tuple(point), 4, color, -1)
                    cv2.putText(display_img, str(j+1), (point[0]+5, point[1]-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                # 兼容旧的矩形模式
                x, y, w, h = area['x'], area['y'], area['width'], area['height']
                scaled_x = int(x * self.scale_factor)
                scaled_y = int(y * self.scale_factor)
                scaled_w = int(w * self.scale_factor)
                scaled_h = int(h * self.scale_factor)
                
                cv2.rectangle(display_img, (scaled_x, scaled_y), 
                             (scaled_x + scaled_w, scaled_y + scaled_h), color, 2)
            
            # 添加镜像区域标签
            label_x = int((area.get('x', 0) if 'points' not in area else area['points'][0][0]) * self.scale_factor)
            label_y = int((area.get('y', 0) if 'points' not in area else area['points'][0][1]) * self.scale_factor)
            cv2.putText(display_img, f"Mirror {i+1}", (label_x + 5, label_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 转换为PhotoImage并显示
        pil_img = Image.fromarray(display_img)
        self.photo = ImageTk.PhotoImage(pil_img)
        
        # 清除画布并绘制图像
        self.canvas.delete("all")
        self.canvas_img = self.canvas.create_image(
            self.canvas_offset_x, self.canvas_offset_y, 
            image=self.photo, anchor="nw"
        )
        
        # 更新滚动区域
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)
    
    def canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> Tuple[int, int]:
        """将画布坐标转换为图像坐标"""
        img_x = int((canvas_x - self.canvas_offset_x) / self.scale_factor)
        img_y = int((canvas_y - self.canvas_offset_y) / self.scale_factor)
        return img_x, img_y
    
    def image_to_canvas_coords(self, img_x: int, img_y: int) -> Tuple[int, int]:
        """将图像坐标转换为画布坐标"""
        canvas_x = int(img_x * self.scale_factor + self.canvas_offset_x)
        canvas_y = int(img_y * self.scale_factor + self.canvas_offset_y)
        return canvas_x, canvas_y
    
    def update_area_list(self):
        """更新区域列表"""
        self.area_listbox.delete(0, tk.END)
        for i, area in enumerate(self.print_areas):
            if 'points' in area:
                # 四点模式
                points_str = ", ".join([f"({p[0]},{p[1]})" for p in area['points']])
                self.area_listbox.insert(tk.END, f"区域 {i+1}: {points_str}")
            else:
                # 矩形模式（兼容）
                self.area_listbox.insert(tk.END, 
                    f"区域 {i+1}: ({area['x']}, {area['y']}) {area['width']}x{area['height']}")
        
        # 更新镜像区域列表
        self.update_mirror_area_list()
    
    def update_mirror_area_list(self):
        """更新镜像区域列表"""
        self.mirror_listbox.delete(0, tk.END)
        for i, area in enumerate(self.mirror_areas):
            if 'points' in area:
                points_str = ", ".join([f"({p[0]},{p[1]})" for p in area['points']])
                self.mirror_listbox.insert(tk.END, f"补充区域 {i+1}: {points_str}")
            else:
                # 兼容旧格式
                self.mirror_listbox.insert(tk.END, 
                    f"补充区域 {i+1}: ({area['x']}, {area['y']}) {area['width']}x{area['height']}")
    
    def on_area_select(self, event):
        """区域选择事件"""
        selection = self.area_listbox.curselection()
        if selection:
            self.current_area = selection[0]
            area = self.print_areas[self.current_area]
            
            if 'points' in area:
                # 四点模式
                points = area['points']
                self.tl_x_var.set(points[0][0])  # 左上
                self.tl_y_var.set(points[0][1])
                self.tr_x_var.set(points[1][0])  # 右上
                self.tr_y_var.set(points[1][1])
                self.br_x_var.set(points[2][0])  # 右下
                self.br_y_var.set(points[2][1])
                self.bl_x_var.set(points[3][0])  # 左下
                self.bl_y_var.set(points[3][1])
            else:
                # 矩形模式（兼容）- 转换为四点
                x, y, w, h = area['x'], area['y'], area['width'], area['height']
                self.tl_x_var.set(x)      # 左上
                self.tl_y_var.set(y)
                self.tr_x_var.set(x + w)  # 右上
                self.tr_y_var.set(y)
                self.br_x_var.set(x + w)  # 右下
                self.br_y_var.set(y + h)
                self.bl_x_var.set(x)      # 左下
                self.bl_y_var.set(y + h)
            
            self.display_template()
    
    def add_area(self):
        """添加打印区域"""
        if self.edit_mode == "rectangle":
            # 矩形模式 - 创建一个标准矩形
            new_area = {
                'points': [[50, 50], [150, 50], [150, 150], [50, 150]]
            }
        else:
            # 自由模式 - 创建一个梯形
            new_area = {
                'points': [[50, 50], [150, 60], [140, 150], [60, 140]]
            }
        
        self.print_areas.append(new_area)
        self.update_area_list()
        self.display_template()
        
        # 选择新添加的区域
        self.area_listbox.selection_set(len(self.print_areas) - 1)
        self.on_area_select(None)
    
    def delete_area(self):
        """删除打印区域"""
        if self.current_area is not None and 0 <= self.current_area < len(self.print_areas):
            self.print_areas.pop(self.current_area)
            self.current_area = None
            self.update_area_list()
            self.display_template()
            
            # 清空输入框
            for var in [self.tl_x_var, self.tl_y_var, self.tr_x_var, self.tr_y_var,
                    self.br_x_var, self.br_y_var, self.bl_x_var, self.bl_y_var]:
                var.set(0)
    
    def update_area(self):
        """更新区域属性"""
        if self.current_area is not None and 0 <= self.current_area < len(self.print_areas):
            area = self.print_areas[self.current_area]
            
            # 更新四个坐标点
            area['points'] = [
                [self.tl_x_var.get(), self.tl_y_var.get()],  # 左上
                [self.tr_x_var.get(), self.tr_y_var.get()],  # 右上
                [self.br_x_var.get(), self.br_y_var.get()],  # 右下
                [self.bl_x_var.get(), self.bl_y_var.get()]   # 左下
            ]
            
            # 移除旧的矩形属性（如果存在）
            for key in ['x', 'y', 'width', 'height']:
                if key in area:
                    del area[key]
            
            self.update_area_list()
            self.display_template()
    
    def start_pick_mirror_point(self, point_type: str):
        """开始拾取镜像区域坐标点"""
        self.picking_point = f"mirror_{point_type}"
        self.area_type = "mirror"
        self.original_cursor = self.canvas.cget("cursor")
        self.canvas.config(cursor="crosshair")
    
    def add_mirror_area(self):
        """添加镜像区域"""
        new_area = {
            'points': [[50, 50], [150, 50], [150, 150], [50, 150]]
        }
        self.mirror_areas.append(new_area)
        self.update_mirror_area_list()
        self.display_template()
        
        # 选择新添加的区域
        self.mirror_listbox.selection_set(len(self.mirror_areas) - 1)
        self.on_mirror_area_select(None)
    
    def delete_mirror_area(self):
        """删除镜像区域"""
        if self.current_mirror_area is not None and 0 <= self.current_mirror_area < len(self.mirror_areas):
            self.mirror_areas.pop(self.current_mirror_area)
            self.current_mirror_area = None
            self.update_mirror_area_list()
            self.display_template()
            
            # 清空输入框
            for var in [self.mirror_tl_x_var, self.mirror_tl_y_var, self.mirror_tr_x_var, self.mirror_tr_y_var,
                       self.mirror_br_x_var, self.mirror_br_y_var, self.mirror_bl_x_var, self.mirror_bl_y_var]:
                var.set(0)
    
    def on_mirror_area_select(self, event):
        """镜像区域选择事件"""
        selection = self.mirror_listbox.curselection()
        if selection:
            self.current_mirror_area = selection[0]
            area = self.mirror_areas[self.current_mirror_area]
            
            if 'points' in area:
                # 四点模式
                points = area['points']
                self.mirror_tl_x_var.set(points[0][0])  # 左上
                self.mirror_tl_y_var.set(points[0][1])
                self.mirror_tr_x_var.set(points[1][0])  # 右上
                self.mirror_tr_y_var.set(points[1][1])
                self.mirror_br_x_var.set(points[2][0])  # 右下
                self.mirror_br_y_var.set(points[2][1])
                self.mirror_bl_x_var.set(points[3][0])  # 左下
                self.mirror_bl_y_var.set(points[3][1])
            else:
                # 矩形模式（兼容）- 转换为四点
                x, y, w, h = area['x'], area['y'], area['width'], area['height']
                self.mirror_tl_x_var.set(x)      # 左上
                self.mirror_tl_y_var.set(y)
                self.mirror_tr_x_var.set(x + w)  # 右上
                self.mirror_tr_y_var.set(y)
                self.mirror_br_x_var.set(x + w)  # 右下
                self.mirror_br_y_var.set(y + h)
                self.mirror_bl_x_var.set(x)      # 左下
                self.mirror_bl_y_var.set(y + h)
            
            self.display_template()
    
    def update_mirror_area(self):
        """更新镜像区域属性"""
        if self.current_mirror_area is not None and 0 <= self.current_mirror_area < len(self.mirror_areas):
            area = self.mirror_areas[self.current_mirror_area]
            
            # 更新四个坐标点
            area['points'] = [
                [self.mirror_tl_x_var.get(), self.mirror_tl_y_var.get()],  # 左上
                [self.mirror_tr_x_var.get(), self.mirror_tr_y_var.get()],  # 右上
                [self.mirror_br_x_var.get(), self.mirror_br_y_var.get()],  # 右下
                [self.mirror_bl_x_var.get(), self.mirror_bl_y_var.get()]   # 左下
            ]
            
            # 移除旧的矩形属性（如果存在）
            for key in ['x', 'y', 'width', 'height']:
                if key in area:
                    del area[key]
            
            self.update_mirror_area_list()
            self.display_template()
    
    def on_canvas_click(self, event):
        """画布点击事件"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # 检查是否在图像范围内
        if (canvas_x < self.canvas_offset_x or canvas_y < self.canvas_offset_y or
            canvas_x > self.canvas_offset_x + self.img_width * self.scale_factor or
            canvas_y > self.canvas_offset_y + self.img_height * self.scale_factor):
            return

        # 如果在坐标拾取模式
        if self.picking_point:
            # 转换为图像坐标
            img_x, img_y = self.canvas_to_image_coords(canvas_x, canvas_y)
            
            # 更新对应的坐标输入框
            if self.picking_point == "tl":
                self.tl_x_var.set(img_x)
                self.tl_y_var.set(img_y)
            elif self.picking_point == "tr":
                self.tr_x_var.set(img_x)
                self.tr_y_var.set(img_y)
            elif self.picking_point == "br":
                self.br_x_var.set(img_x)
                self.br_y_var.set(img_y)
            elif self.picking_point == "bl":
                self.bl_x_var.set(img_x)
                self.bl_y_var.set(img_y)
            # 镜像区域坐标拾取
            elif self.picking_point == "mirror_tl":
                self.mirror_tl_x_var.set(img_x)
                self.mirror_tl_y_var.set(img_y)
            elif self.picking_point == "mirror_tr":
                self.mirror_tr_x_var.set(img_x)
                self.mirror_tr_y_var.set(img_y)
            elif self.picking_point == "mirror_br":
                self.mirror_br_x_var.set(img_x)
                self.mirror_br_y_var.set(img_y)
            elif self.picking_point == "mirror_bl":
                self.mirror_bl_x_var.set(img_x)
                self.mirror_bl_y_var.set(img_y)
            
            self.end_pick_point()
            return

        if self.edit_mode == "free" and self.point_picking_sequence is not None:
            # 转换为图像坐标
            img_x, img_y = self.canvas_to_image_coords(canvas_x, canvas_y)
            
            # 如果是新区域的第一个点，创建新区域
            if self.point_picking_sequence == 0:
                new_area = {
                    'points': [[0, 0], [0, 0], [0, 0], [0, 0]]  # 初始化四个点
                }
                self.print_areas.append(new_area)
                self.current_area = len(self.print_areas) - 1
                self.area_listbox.selection_clear(0, tk.END)
                self.area_listbox.selection_set(self.current_area)
            
            # 更新当前点的坐标
            area = self.print_areas[self.current_area]
            area['points'][self.point_picking_sequence] = [img_x, img_y]
            
            # 更新对应的输入框
            if self.point_picking_sequence == 0:    # 左上
                self.tl_x_var.set(img_x)
                self.tl_y_var.set(img_y)
            elif self.point_picking_sequence == 1:  # 右上
                self.tr_x_var.set(img_x)
                self.tr_y_var.set(img_y)
            elif self.point_picking_sequence == 2:  # 右下
                self.br_x_var.set(img_x)
                self.br_y_var.set(img_y)
            elif self.point_picking_sequence == 3:  # 左下
                self.bl_x_var.set(img_x)
                self.bl_y_var.set(img_y)
            
            # 更新显示
            self.display_template()
            # 移动到下一个点
            self.point_picking_sequence += 1
            if self.point_picking_sequence > 3:
                # 完成所有点的拾取
                self.point_picking_sequence = None
                self.update_area_list()
                messagebox.showinfo("提示", "四个角点设置完成")

        elif self.edit_mode == "rectangle":
            # 矩形模式：开始拖拽创建矩形
            self.drag_start = (canvas_x, canvas_y)
        else:        
            messagebox.showinfo("提示", "拾取模式出错")
    
    def on_canvas_drag(self, event):
        """画布拖拽事件"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        if self.edit_mode == "rectangle" and self.drag_start:
            # 矩形模式：绘制拖拽矩形
            if self.drag_rect:
                self.canvas.delete(self.drag_rect)
            
            self.drag_rect = self.canvas.create_rectangle(
                self.drag_start[0], self.drag_start[1], canvas_x, canvas_y,
                outline="red", width=2, dash=(5, 5))
    
    def on_canvas_release(self, event):
        """画布释放事件"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        if self.edit_mode == "rectangle" and self.drag_start and self.drag_rect:
            # 矩形模式：完成矩形创建
            x1, y1 = self.drag_start
            x2, y2 = canvas_x, canvas_y
            
            # 确保坐标正确
            left, top = min(x1, x2), min(y1, y2)
            right, bottom = max(x1, x2), max(y1, y2)
            
            if (right - left) > 10 and (bottom - top) > 10:  # 最小区域大小
                # 转换为图像坐标
                img_left, img_top = self.canvas_to_image_coords(left, top)
                img_right, img_bottom = self.canvas_to_image_coords(right, bottom)
                
                # 添加新的打印区域（四点格式）
                new_area = {
                    'points': [
                        [img_left, img_top],      # 左上
                        [img_right, img_top],     # 右上
                        [img_right, img_bottom],  # 右下
                        [img_left, img_bottom]    # 左下
                    ]
                }
                
                self.print_areas.append(new_area)
                self.update_area_list()
                self.display_template()
                
                # 选择新添加的区域
                self.area_listbox.selection_set(len(self.print_areas) - 1)
                self.on_area_select(None)
            
            self.canvas.delete(self.drag_rect)
            self.drag_start = None
            self.drag_rect = None
    
    def on_right_click(self, event):
        """右键点击事件 - 开始平移"""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.last_canvas_x = self.canvas_offset_x
        self.last_canvas_y = self.canvas_offset_y
        self.canvas.config(cursor="fleur")
    
    def on_right_drag(self, event):
        """右键拖拽事件 - 平移图像"""
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            self.canvas_offset_x = self.last_canvas_x + dx
            self.canvas_offset_y = self.last_canvas_y + dy
            
            self.display_template()
    
    def on_right_release(self, event):
        """右键释放事件 - 结束平移"""
        self.is_panning = False
        self.canvas.config(cursor="")
    
    def on_mouse_wheel(self, event):
        """鼠标滚轮事件 - 缩放"""
        # 获取鼠标在画布上的位置
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # 计算缩放前鼠标在图像上的位置
        old_img_x = (canvas_x - self.canvas_offset_x) / self.scale_factor
        old_img_y = (canvas_y - self.canvas_offset_y) / self.scale_factor
        
        # 确定缩放方向
        if event.delta > 0 or event.num == 4:
            # 向上滚动 - 放大
            scale_change = 1.1
        else:
            # 向下滚动 - 缩小
            scale_change = 0.9
        
        # 更新缩放因子
        new_scale = self.scale_factor * scale_change
        if 0.1 <= new_scale <= 10.0:  # 限制缩放范围
            self.scale_factor = new_scale
            
            # 调整偏移以保持鼠标位置不变
            new_canvas_x = old_img_x * self.scale_factor + self.canvas_offset_x
            new_canvas_y = old_img_y * self.scale_factor + self.canvas_offset_y
            
            self.canvas_offset_x += canvas_x - new_canvas_x
            self.canvas_offset_y += canvas_y - new_canvas_y
            
            self.update_zoom_label()
            self.display_template()
    
    def select_nearest_point(self, canvas_x: float, canvas_y: float):
        """选择最近的区域角点"""
        if not self.print_areas:
            return
        
        min_distance = float('inf')
        nearest_area = None
        nearest_point = None
        
        for i, area in enumerate(self.print_areas):
            if 'points' in area:
                for j, point in enumerate(area['points']):
                    # 转换点坐标到画布坐标
                    point_canvas_x, point_canvas_y = self.image_to_canvas_coords(point[0], point[1])
                    
                    # 计算距离
                    distance = math.sqrt((canvas_x - point_canvas_x)**2 + (canvas_y - point_canvas_y)**2)
                    
                    if distance < min_distance and distance < 20:  # 20像素内算作点击
                        min_distance = distance
                        nearest_area = i
                        nearest_point = j
        
        if nearest_area is not None:
            # 选择该区域
            self.current_area = nearest_area
            self.area_listbox.selection_clear(0, tk.END)
            self.area_listbox.selection_set(nearest_area)
            self.on_area_select(None)
            
            # 高亮选中的点
            self.highlight_selected_point(nearest_area, nearest_point)
        """画布释放事件"""
    def highlight_selected_point(self, area_index: int, point_index: int):
        """高亮选中的点"""
        # 这个方法可以在将来扩展，用于高亮显示选中的角点
        pass
    
    def confirm(self):
        """确认并保存"""
        name = self.name_var.get().strip()
        category = self.category_var.get().strip()
        
        if not name:
            messagebox.showerror("错误", "请输入模板名称")
            return
        
        if not category:
            messagebox.showerror("错误", "请选择或输入分类")
            return
        
        if not self.print_areas:
            messagebox.showerror("错误", "请至少添加一个打印区域")
            return
        
        # 转换打印区域格式以兼容旧系统
        converted_areas = []
        for area in self.print_areas:
            if 'points' in area:
                # 四点模式 - 保留points，同时计算兼容的矩形属性
                points = area['points']
                min_x = min(p[0] for p in points)
                min_y = min(p[1] for p in points)
                max_x = max(p[0] for p in points)
                max_y = max(p[1] for p in points)
                
                converted_area = {
                    'points': points,  # 新的四点格式
                    'x': min_x,       # 兼容旧格式
                    'y': min_y,
                    'width': max_x - min_x,
                    'height': max_y - min_y
                }
            else:
                # 旧矩形模式 - 转换为四点格式
                x, y, w, h = area['x'], area['y'], area['width'], area['height']
                converted_area = {
                    'points': [[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                }
            
            converted_areas.append(converted_area)
        
        # 转换镜像区域格式
        converted_mirror_areas = []
        for area in self.mirror_areas:
            if 'points' in area:
                converted_mirror_areas.append({
                    'points': area['points']
                })
            else:
                # 兼容旧格式
                x, y, w, h = area['x'], area['y'], area['width'], area['height']
                converted_mirror_areas.append({
                    'points': [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                })

        self.result = {
            'name': name,
            'category': category,
            'print_areas': converted_areas,
            'mirror_areas': converted_mirror_areas
        }
        
        self.dialog.destroy()
    
    def cancel(self):
        """取消"""
        self.result = None
        self.dialog.destroy()
