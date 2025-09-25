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


class ImageProcessor:
    """图像处理类"""
    
    @staticmethod
    def perspective_transform(image: np.ndarray, src_points: np.ndarray, 
                            dst_points: np.ndarray) -> np.ndarray:
        """透视变换"""
        matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), 
                                           dst_points.astype(np.float32))
        height, width = dst_points[:, 1].max() - dst_points[:, 1].min(), \
                       dst_points[:, 0].max() - dst_points[:, 0].min()
        return cv2.warpPerspective(image, matrix, (int(width), int(height)))
    
    @staticmethod
    def add_shadow(image: np.ndarray, offset: Tuple[int, int] = (5, 5), 
                  blur_radius: int = 10, shadow_color: Tuple[int, int, int] = (0, 0, 0),
                  shadow_opacity: float = 0.5) -> np.ndarray:
        """添加阴影效果"""
        if image.shape[2] == 3:
            # 如果是RGB图像，添加alpha通道
            alpha = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
            image = np.concatenate([image, alpha], axis=2)
        
        # 创建阴影
        shadow = np.zeros_like(image)
        shadow[:, :, :3] = shadow_color
        shadow[:, :, 3] = (image[:, :, 3] * shadow_opacity).astype(image.dtype)
        
        # 偏移阴影
        shadow_shifted = np.zeros_like(shadow)
        h, w = shadow.shape[:2]
        if offset[1] > 0 and offset[0] > 0:
            shadow_shifted[offset[1]:, offset[0]:] = shadow[:-offset[1], :-offset[0]]
        
        # 模糊阴影
        if blur_radius > 0:
            shadow_shifted = cv2.GaussianBlur(shadow_shifted, (blur_radius*2+1, blur_radius*2+1), 0)
        
        # 合成图像
        result = np.zeros((h + abs(offset[1]), w + abs(offset[0]), 4), dtype=image.dtype)
        
        # 放置阴影
        result[:shadow_shifted.shape[0], :shadow_shifted.shape[1]] = shadow_shifted
        
        # 放置原图像
        start_y = max(0, -offset[1])
        start_x = max(0, -offset[0])
        result[start_y:start_y+h, start_x:start_x+w] = ImageProcessor.blend_alpha(
            result[start_y:start_y+h, start_x:start_x+w], image)
        
        return result
    
    @staticmethod
    def blend_alpha(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
        """Alpha通道混合"""
        if background.shape != foreground.shape:
            return foreground
        
        alpha_f = foreground[:, :, 3:4] / 255.0
        alpha_b = background[:, :, 3:4] / 255.0
        alpha_out = alpha_f + alpha_b * (1 - alpha_f)
        
        result = np.zeros_like(foreground)
        mask = alpha_out[:, :, 0] > 0  # 修改这里，将掩码转换为2D
        
        # 分别处理RGB通道和Alpha通道
        for i in range(3):  # RGB通道
            result[:, :, i][mask] = (
                foreground[:, :, i][mask] * alpha_f[:, :, 0][mask] + 
                background[:, :, i][mask] * alpha_b[:, :, 0][mask] * (1 - alpha_f[:, :, 0][mask])
            ) / alpha_out[:, :, 0][mask]
        
        # 处理Alpha通道
        result[:, :, 3] = (alpha_out * 255).astype(foreground.dtype)[:, :, 0]
        
        return result
    
    @staticmethod
    def apply_blur(image: np.ndarray, blur_radius: int) -> np.ndarray:
        """应用模糊效果"""
        if blur_radius <= 0:
            return image
        return cv2.GaussianBlur(image, (blur_radius*2+1, blur_radius*2+1), 0)
    
    @staticmethod
    def to_grayscale(image: np.ndarray, preserve_alpha: bool = True) -> np.ndarray:
        """转换为灰度图像"""
        if len(image.shape) == 3:
            if image.shape[2] == 4 and preserve_alpha:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
                return np.dstack([gray, gray, gray, image[:, :, 3]])
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return np.dstack([gray, gray, gray])
        return image
    
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int], 
                               keep_aspect: bool = True) -> np.ndarray:
        """按比例调整图像大小"""
        if not keep_aspect:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 创建目标尺寸的画布
        if len(image.shape) == 3:
            canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # 居中放置
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return canvas

