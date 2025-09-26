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
        # 计算目标区域的边界
        min_x = dst_points[:, 0].min()
        max_x = dst_points[:, 0].max()
        min_y = dst_points[:, 1].min()
        max_y = dst_points[:, 1].max()
        
        # 计算输出图像尺寸
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        
        # 调整目标点坐标，使其相对于输出图像的原点
        adjusted_dst_points = dst_points.copy()
        adjusted_dst_points[:, 0] -= min_x
        adjusted_dst_points[:, 1] -= min_y
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), 
                                           adjusted_dst_points.astype(np.float32))
        
        # 执行透视变换
        return cv2.warpPerspective(image, matrix, (width, height))
    
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

    @staticmethod
    def mirror_fill_area(image: np.ndarray, area_points: List[List[int]], source_image: np.ndarray = None) -> np.ndarray:
        """
        透视变换镜像填充指定区域
        
        Args:
            image: 输入图像（模板）
            area_points: 区域的四个角点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            source_image: 源图像（图案内容），如果为None则使用输入图像
            
        Returns:
            填充后的图像
        """
        if len(area_points) != 4:
            return image
            
        result = image.copy()
        h, w = image.shape[:2]
        
        # 如果没有提供源图像，使用输入图像
        if source_image is None:
            source_image = image

        # 创建区域掩码
        points = np.array(area_points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        # 获取区域边界框
        x_coords = [p[0] for p in area_points]
        y_coords = [p[1] for p in area_points]
        min_x, max_x = max(0, min(x_coords)), min(w-1, max(x_coords))
        min_y, max_y = max(0, min(y_coords)), min(h-1, max(y_coords))
        
        # 计算区域尺寸
        area_width = max_x - min_x + 1
        area_height = max_y - min_y + 1
        
        # 获取源图像尺寸
        src_h, src_w = source_image.shape[:2]
        
        print(f"透视镜像填充: 区域尺寸 {area_width}x{area_height}, 位置 ({min_x},{min_y})-({max_x},{max_y})")
        print(f"模板尺寸: {w}x{h}, 图案尺寸: {src_w}x{src_h}")
        
        try:
            # 步骤1: 根据区域大小裁剪源图像
            # 计算裁剪区域，保持宽高比
            aspect_ratio = area_width / area_height
            src_aspect_ratio = src_w / src_h
            
            if src_aspect_ratio > aspect_ratio:
                # 源图像更宽，按高度裁剪
                crop_height = src_h
                crop_width = int(crop_height * aspect_ratio)
                crop_x = max(0, (src_w - crop_width) // 2)
                crop_y = 0
            else:
                # 源图像更高，按宽度裁剪
                crop_width = src_w
                crop_height = int(crop_width / aspect_ratio)
                crop_x = 0
                crop_y = max(0, (src_h - crop_height) // 2)
            
            # 确保裁剪区域在源图像范围内
            crop_x = max(0, min(crop_x, src_w - 1))
            crop_y = max(0, min(crop_y, src_h - 1))
            crop_width = min(crop_width, src_w - crop_x)
            crop_height = min(crop_height, src_h - crop_y)
            
            # 裁剪源图像
            cropped_source = source_image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
            
            print(f"裁剪源图像: ({crop_x},{crop_y}) 尺寸 {crop_width}x{crop_height}")
            
            # 步骤2: 对裁剪的图像进行镜像处理
            mirrored_source = cv2.flip(cropped_source, 1)  # 水平镜像
            
            # 步骤3: 将镜像图像缩放到区域大小
            resized_mirror = cv2.resize(mirrored_source, (area_width, area_height), interpolation=cv2.INTER_LINEAR)
            
            print(f"图像格式调试: 原图形状 {image.shape}, 裁剪后 {cropped_source.shape}, 镜像后 {mirrored_source.shape}, 缩放后 {resized_mirror.shape}")
            
            # 步骤4: 使用透视变换将镜像图像映射到坐标区域
            # 定义源点（矩形区域的四个角点）
            src_points = np.float32([
                [0, 0],                           # 左上
                [area_width - 1, 0],              # 右上
                [area_width - 1, area_height - 1], # 右下
                [0, area_height - 1]              # 左下
            ])
            
            # 目标点（实际的区域坐标，相对于边界框）
            dst_points = np.float32([
                [area_points[0][0] - min_x, area_points[0][1] - min_y],  # 左上
                [area_points[1][0] - min_x, area_points[1][1] - min_y],  # 右上
                [area_points[2][0] - min_x, area_points[2][1] - min_y],  # 右下
                [area_points[3][0] - min_x, area_points[3][1] - min_y]   # 左下
            ])
            
            # 计算透视变换矩阵
            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # 应用透视变换
            transformed_mirror = cv2.warpPerspective(
                resized_mirror, 
                perspective_matrix, 
                (area_width, area_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )
            
            print(f"透视变换完成，变换矩阵形状: {perspective_matrix.shape}")
            print(f"变换后图像形状: {transformed_mirror.shape}, 数据类型: {transformed_mirror.dtype}")
            
            # 步骤5: 将变换后的镜像图像应用到结果图像
            for y in range(area_height):
                for x in range(area_width):
                    img_y = min_y + y
                    img_x = min_x + x
                    if (0 <= img_y < h and 0 <= img_x < w and 
                        mask[img_y, img_x] > 0 and 
                        y < transformed_mirror.shape[0] and x < transformed_mirror.shape[1]):
                        
                        # 获取变换后的像素值
                        pixel_value = transformed_mirror[y, x]
                        
                        # 检查像素是否有效（非零或非透明）
                        is_valid_pixel = False
                        
                        if len(transformed_mirror.shape) == 3:
                            if transformed_mirror.shape[2] == 4:  # RGBA
                                if pixel_value[3] > 0:  # 非透明
                                    pixel_value = pixel_value[:3]  # 只取RGB
                                    is_valid_pixel = True
                            else:  # RGB
                                if np.any(pixel_value > 0):  # 非全黑
                                    is_valid_pixel = True
                        else:  # 灰度
                            if pixel_value > 0:
                                is_valid_pixel = True
                        
                        # 确保像素值与结果图像通道数匹配
                        if is_valid_pixel:
                            try:
                                # 获取结果图像和像素值的维度信息
                                result_channels = result.shape[2] if len(result.shape) == 3 else 1
                                pixel_channels = len(pixel_value) if hasattr(pixel_value, '__len__') else 1
                                
                                if result_channels == pixel_channels:
                                    # 通道数匹配，直接赋值
                                    result[img_y, img_x] = pixel_value
                                elif result_channels == 3 and pixel_channels == 1:
                                    # 结果是RGB，像素是灰度
                                    result[img_y, img_x] = [pixel_value, pixel_value, pixel_value]
                                elif result_channels == 4 and pixel_channels == 3:
                                    # 结果是RGBA，像素是RGB
                                    result[img_y, img_x] = np.append(pixel_value, 255)
                                elif result_channels == 1 and pixel_channels == 3:
                                    # 结果是灰度，像素是RGB
                                    gray_value = np.mean(pixel_value)
                                    result[img_y, img_x] = gray_value
                                elif result_channels == 3 and pixel_channels >= 3:
                                    # 结果是RGB，像素有更多通道
                                    result[img_y, img_x] = pixel_value[:3]
                                else:
                                    # 其他情况，尝试强制转换
                                    if result_channels == 3:
                                        if isinstance(pixel_value, (int, float, np.integer, np.floating)):
                                            result[img_y, img_x] = [pixel_value, pixel_value, pixel_value]
                                        else:
                                            result[img_y, img_x] = pixel_value[:3] if len(pixel_value) >= 3 else [pixel_value[0], pixel_value[0], pixel_value[0]]
                                    else:
                                        result[img_y, img_x] = pixel_value
                                        
                            except (ValueError, IndexError) as e:
                                # 如果赋值失败，跳过这个像素
                                print(f"像素赋值失败 ({img_x},{img_y}): {e}, 像素值: {pixel_value}, 结果形状: {result.shape}")
                                continue
            
            print("透视镜像填充完成")
            
        except Exception as e:
            print(f"透视镜像填充出错: {e}, 使用边缘颜色填充")
            # 如果出错，使用边缘颜色填充
            border_mask = cv2.dilate(mask, np.ones((3,3), np.uint8)) - mask
            if np.sum(border_mask) > 0:
                if len(image.shape) == 3:
                    avg_color = np.mean(image[border_mask > 0], axis=0)
                    result[mask > 0] = avg_color
                else:
                    avg_color = np.mean(image[border_mask > 0])
                    result[mask > 0] = avg_color
        
        return result
    
    @staticmethod
    def smart_mirror_fill(image: np.ndarray, area_points: List[List[int]], source_image: np.ndarray = None) -> np.ndarray:
        """
        智能镜像填充 - 使用简化的镜像填充算法
        
        Args:
            image: 输入图像（模板）
            area_points: 区域的四个角点坐标
            source_image: 源图像（图案内容），如果为None则使用输入图像
            
        Returns:
            填充后的图像
        """
        # 直接调用简化的镜像填充方法
        return ImageProcessor.mirror_fill_area(image, area_points, source_image)

