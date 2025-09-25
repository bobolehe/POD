import tkinter as tk
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

class BatchGenerateDialog:
    """批量生成对话框"""
    
    def __init__(self, parent, templates: List[Dict], patterns: List[Dict]):
        self.result = None
        self.templates = templates
        self.patterns = patterns
        
        # 创建对话框窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("批量生成设置")
        # 设置窗口大小
        w, h = 600, 500
        # 获取屏幕尺寸
        sw = self.dialog.winfo_screenwidth()
        sh = self.dialog.winfo_screenheight()
        # 计算居中位置
        x = (sw - w) // 2
        y = (sh - h) // 2
        # 设置大小和位置
        self.dialog.geometry(f"{w}x{h}+{x}+{y}")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_interface()
        
        # 等待对话框关闭
        self.dialog.wait_window()
    
    def create_interface(self):
        """创建对话框界面"""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 模板选择区域
        template_frame = ttk.LabelFrame(main_frame, text="选择模板", padding=10)
        template_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 全选/全不选按钮
        template_btn_frame = ttk.Frame(template_frame)
        template_btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(template_btn_frame, text="全选", 
                  command=self.select_all_templates).pack(side=tk.LEFT)
        ttk.Button(template_btn_frame, text="全不选", 
                  command=self.deselect_all_templates).pack(side=tk.LEFT, padx=(5, 0))
        
        # 模板列表
        template_list_frame = ttk.Frame(template_frame)
        template_list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.template_vars = {}
        self.template_checkboxes = {}
        
        # 创建滚动框架
        template_canvas = tk.Canvas(template_list_frame, height=150)
        template_scrollbar = ttk.Scrollbar(template_list_frame, orient="vertical", 
                                         command=template_canvas.yview)
        template_scrollable_frame = ttk.Frame(template_canvas)
        
        template_scrollable_frame.bind(
            "<Configure>",
            lambda e: template_canvas.configure(scrollregion=template_canvas.bbox("all"))
        )
        
        template_canvas.create_window((0, 0), window=template_scrollable_frame, anchor="nw")
        template_canvas.configure(yscrollcommand=template_scrollbar.set)
        
        # 添加模板复选框
        for i, template in enumerate(self.templates):
            var = tk.BooleanVar(value=True)
            self.template_vars[template['id']] = var
            
            cb = ttk.Checkbutton(template_scrollable_frame, 
                               text=f"{template['category']} - {template['name']}",
                               variable=var)
            cb.pack(anchor="w", pady=1)
            self.template_checkboxes[template['id']] = cb
        
        template_canvas.pack(side="left", fill="both", expand=True)
        template_scrollbar.pack(side="right", fill="y")
        
        # 图案选择区域
        pattern_frame = ttk.LabelFrame(main_frame, text="选择图案", padding=10)
        pattern_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 全选/全不选按钮
        pattern_btn_frame = ttk.Frame(pattern_frame)
        pattern_btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(pattern_btn_frame, text="全选", 
                  command=self.select_all_patterns).pack(side=tk.LEFT)
        ttk.Button(pattern_btn_frame, text="全不选", 
                  command=self.deselect_all_patterns).pack(side=tk.LEFT, padx=(5, 0))
        
        # 图案列表
        pattern_list_frame = ttk.Frame(pattern_frame)
        pattern_list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.pattern_vars = {}
        self.pattern_checkboxes = {}
        
        # 创建滚动框架
        pattern_canvas = tk.Canvas(pattern_list_frame, height=150)
        pattern_scrollbar = ttk.Scrollbar(pattern_list_frame, orient="vertical",
                                        command=pattern_canvas.yview)
        pattern_scrollable_frame = ttk.Frame(pattern_canvas)
        
        pattern_scrollable_frame.bind(
            "<Configure>",
            lambda e: pattern_canvas.configure(scrollregion=pattern_canvas.bbox("all"))
        )
        
        pattern_canvas.create_window((0, 0), window=pattern_scrollable_frame, anchor="nw")
        pattern_canvas.configure(yscrollcommand=pattern_scrollbar.set)
        
        # 添加图案复选框
        for i, pattern in enumerate(self.patterns):
            var = tk.BooleanVar(value=True)
            self.pattern_vars[i] = var
            
            cb = ttk.Checkbutton(pattern_scrollable_frame, 
                               text=pattern['name'],
                               variable=var)
            cb.pack(anchor="w", pady=1)
            self.pattern_checkboxes[i] = cb
        
        pattern_canvas.pack(side="left", fill="both", expand=True)
        pattern_scrollbar.pack(side="right", fill="y")
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="开始生成", command=self.confirm).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="取消", command=self.cancel).pack(side=tk.RIGHT, padx=(0, 10))
        
        # 统计信息
        self.info_label = ttk.Label(button_frame, text="")
        self.info_label.pack(side=tk.LEFT)
        self.update_info()
        
        # 绑定变量变化事件
        for var in self.template_vars.values():
            var.trace('w', self.on_selection_change)
        for var in self.pattern_vars.values():
            var.trace('w', self.on_selection_change)
    
    def select_all_templates(self):
        """全选模板"""
        for var in self.template_vars.values():
            var.set(True)
    
    def deselect_all_templates(self):
        """全不选模板"""
        for var in self.template_vars.values():
            var.set(False)
    
    def select_all_patterns(self):
        """全选图案"""
        for var in self.pattern_vars.values():
            var.set(True)
    
    def deselect_all_patterns(self):
        """全不选图案"""
        for var in self.pattern_vars.values():
            var.set(False)
    
    def on_selection_change(self, *args):
        """选择变化事件"""
        self.update_info()
    
    def update_info(self):
        """更新统计信息"""
        selected_templates = sum(1 for var in self.template_vars.values() if var.get())
        selected_patterns = sum(1 for var in self.pattern_vars.values() if var.get())
        total_combinations = selected_templates * selected_patterns
        
        self.info_label.config(text=f"将生成 {total_combinations} 个预览图 "
                                  f"({selected_templates} 模板 × {selected_patterns} 图案)")
    
    def confirm(self):
        """确认生成"""
        selected_templates = []
        for template in self.templates:
            if self.template_vars[template['id']].get():
                selected_templates.append(template)
        
        selected_patterns = []
        for i, pattern in enumerate(self.patterns):
            if self.pattern_vars[i].get():
                selected_patterns.append(pattern)
        
        if not selected_templates:
            messagebox.showerror("错误", "请至少选择一个模板")
            return
        
        if not selected_patterns:
            messagebox.showerror("错误", "请至少选择一个图案")
            return
        
        self.result = {
            'templates': selected_templates,
            'patterns': selected_patterns
        }
        
        self.dialog.destroy()
    
    def cancel(self):
        """取消"""
        self.result = None
        self.dialog.destroy()
