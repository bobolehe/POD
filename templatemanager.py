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
from datadasemanager import DatabaseManager
from tkinter import simpledialog


class TemplateManager:
    """模板管理类"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.templates = []
        self.load_templates()
        self.create_default_templates()
    
    def load_templates(self):
        """从数据库加载模板"""
        self.templates = self.db_manager.load_templates()
    
    def create_default_templates(self):
        """创建默认模板"""
        default_templates = [
            {
                'name': '抱枕',
                'category': '家居',
                'size': (400, 400),
                'print_areas': [{'x': 50, 'y': 50, 'width': 300, 'height': 300, 'perspective_points': [[50, 50], [350, 50], [350, 350], [50, 350]]}],
                'color': (250, 250, 250)
            },
            {
                'name': 'A4海报',
                'category': '印刷品',
                'size': (297, 420),
                'print_areas': [{'x': 20, 'y': 20, 'width': 257, 'height': 380, 'perspective_points': [[20, 20], [277, 20], [277, 400], [20, 400]]}],
                'color': (255, 255, 255)
            }
        ]
        
        existing_names = {t['name'] for t in self.templates}
        
        for template_data in default_templates:
            if template_data['name'] not in existing_names:
                # 创建模板图像
                img = self.create_template_image(template_data)
                img_path = f"templates/{template_data['name'].replace(' ', '_')}.png"
                # 确保目录存在
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                # 新建img_path文件
                cv2.imwrite(img_path, img)
                # 保存到数据库
                self.db_manager.save_template(
                    template_data['name'],
                    template_data['category'],
                    img_path,
                    template_data['print_areas']
                )
        self.load_templates()  # 重新加载模板
    
    def create_template_image(self, template_data: Dict) -> np.ndarray:
        """创建模板图像"""
        width, height = template_data['size']
        color = template_data['color']
        
        # 创建基础图像
        img = np.full((height, width, 3), color, dtype=np.uint8)
        
        # 绘制打印区域边框
        for area in template_data['print_areas']:
            cv2.rectangle(img, 
                         (area['x'], area['y']), 
                         (area['x'] + area['width'], area['y'] + area['height']),
                         (200, 200, 200), 2)
            
            # 添加区域标识
            cv2.putText(img, 'Print Area', 
                       (area['x'] + 10, area['y'] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return img
    
    def get_templates_by_category(self) -> Dict[str, List[Dict]]:
        """按分类获取模板"""
        categories = {}
        for template in self.templates:
            category = template['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(template)
        return categories
    
    def add_custom_template(self, name: str, category: str, image_path: str,
                          print_areas: List[Dict]) -> bool:
        """添加自定义模板"""
        return self.db_manager.save_template(name, category, image_path, print_areas)
