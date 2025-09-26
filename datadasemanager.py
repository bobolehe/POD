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


class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self, db_path: str = "pod_templates.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                image_path TEXT NOT NULL,
                print_areas TEXT NOT NULL,
                mirror_areas TEXT,
                extend_areas TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_template(self, name: str, category: str, image_path: str, print_areas: List[Dict], mirror_areas: List[Dict] = None, extend_areas: List[Dict] = None):
        """保存模板到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO templates 
                (name, category, image_path, print_areas, mirror_areas, extend_areas)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, category, image_path, 
                  json.dumps(print_areas or []),
                  json.dumps(mirror_areas or []),
                  json.dumps(extend_areas or [])))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"保存模板失败: {e}")
            return False
        finally:
            conn.close()
    
    def load_templates(self) -> List[Dict]:
        """加载所有模板"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM templates ORDER BY category, name')
        rows = cursor.fetchall()
        conn.close()
        
        templates = []
        for row in rows:
            templates.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'image_path': row[3],
                'print_areas': json.loads(row[4]),
                'mirror_areas': json.loads(row[5] or '[]'),
                'extend_areas': json.loads(row[6] or '[]'),
                'created_at': row[7]
            })
        
        return templates
    
    def delete_template(self, template_id: int):
        """删除模板"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM templates WHERE id = ?', (template_id,))
        conn.commit()
        conn.close()
