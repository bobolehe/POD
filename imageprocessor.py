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
        
        # 将点坐标转换为numpy数组
        points = np.array(area_points, dtype=np.float32)
        
        # 确保所有点都在图像范围内
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)
        
        # 获取区域边界框
        min_x = int(np.min(points[:, 0]))
        max_x = int(np.max(points[:, 0]))
        min_y = int(np.min(points[:, 1]))
        max_y = int(np.max(points[:, 1]))
        
        # 计算区域尺寸
        area_width = max_x - min_x + 1
        area_height = max_y - min_y + 1
        
        print(f"镜像填充区域: 尺寸 {area_width}x{area_height}, 位置 ({min_x},{min_y})-({max_x},{max_y})")
        
        try:
            # 智能判断镜像方向和取样位置
            # 通过分析四个角点的位置关系判断这是哪个边的镜像区域
            
            # 计算区域的中心点
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            
            # 判断是左侧还是右侧的镜像区域
            # 通常书脊在左侧，所以镜像区域在主图案的左边
            is_left_side = center_x < w / 3  # 如果中心在左1/3区域，认为是左侧镜像
            
            # 查找主要的打印区域（通常是最大的那个）
            main_print_area = None
            if hasattr(image, 'main_print_area_bounds'):
                # 如果有预先定义的主打印区域
                main_print_area = image.main_print_area_bounds
            else:
                # 尝试自动检测主打印区域
                # 假设主打印区域在图像的右侧2/3部分
                main_area_left = int(w * 0.3)
                main_area_right = w - 50  # 留一些边距
                main_area_top = 50
                main_area_bottom = h - 50
            
            # 从结果图像中取样（已经包含了图案）
            if source_image is not None and source_image.shape[:2] != image.shape[:2]:
                # 如果提供了源图像且尺寸不同，使用源图像
                sample_source = source_image
            else:
                # 使用当前的结果图像（已包含图案）
                sample_source = result
            
            # 根据镜像区域的位置，从主图案的边缘取样
            if is_left_side:
                # 左侧镜像区域：取主图案的左边缘
                # 计算取样宽度（通常取镜像区域宽度的2-3倍，以获得更多细节）
                sample_width = min(area_width * 2, int(w * 0.1))  # 最多取图像宽度的10%
                
                # 找到主图案的左边缘位置
                # 通过寻找非背景色的最左边界
                sample_left = main_area_left if 'main_area_left' in locals() else int(w * 0.3)
                sample_right = sample_left + sample_width
                
                # 取样区域：主图案的左边缘
                if sample_right <= sample_source.shape[1]:
                    sampled_edge = sample_source[min_y:max_y, sample_left:sample_right].copy()
                else:
                    sampled_edge = sample_source[min_y:max_y, :area_width].copy()
                
                print(f"从位置 ({sample_left}, {min_y}) 取样，尺寸 {sampled_edge.shape}")
                
                # 水平镜像
                mirrored = cv2.flip(sampled_edge, 1)
                
            else:
                # 右侧镜像区域：取主图案的右边缘
                sample_width = min(area_width * 2, int(w * 0.1))
                
                # 找到主图案的右边缘位置
                sample_right = main_area_right if 'main_area_right' in locals() else w - 50
                sample_left = max(0, sample_right - sample_width)
                
                # 取样区域：主图案的右边缘
                if sample_left >= 0:
                    sampled_edge = sample_source[min_y:max_y, sample_left:sample_right].copy()
                else:
                    sampled_edge = sample_source[min_y:max_y, -area_width:].copy()
                
                print(f"从位置 ({sample_left}, {min_y}) 取样，尺寸 {sampled_edge.shape}")
                
                # 水平镜像
                mirrored = cv2.flip(sampled_edge, 1)
            
            # 调整镜像图像到目标区域大小
            if mirrored.shape[:2] != (area_height, area_width):
                mirrored = cv2.resize(mirrored, (area_width, area_height), 
                                    interpolation=cv2.INTER_LINEAR)
            
            # 确保图像格式匹配
            if len(result.shape) != len(mirrored.shape):
                if len(result.shape) == 3 and len(mirrored.shape) == 2:
                    mirrored = cv2.cvtColor(mirrored, cv2.COLOR_GRAY2BGR)
                elif len(result.shape) == 2 and len(mirrored.shape) == 3:
                    mirrored = cv2.cvtColor(mirrored, cv2.COLOR_BGR2GRAY)
            elif len(result.shape) == 3 and len(mirrored.shape) == 3:
                if result.shape[2] != mirrored.shape[2]:
                    if result.shape[2] == 3 and mirrored.shape[2] == 4:
                        mirrored = mirrored[:, :, :3]
                    elif result.shape[2] == 4 and mirrored.shape[2] == 3:
                        alpha = np.ones((mirrored.shape[0], mirrored.shape[1], 1), 
                                    dtype=mirrored.dtype) * 255
                        mirrored = np.concatenate([mirrored, alpha], axis=2)
            
            # 应用透视变换
            # 源点（镜像图像的四个角）
            src_points = np.float32([
                [0, 0],
                [area_width - 1, 0],
                [area_width - 1, area_height - 1],
                [0, area_height - 1]
            ])
            
            # 目标点
            dst_points = points.astype(np.float32)
            
            # 计算透视变换矩阵
            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # 应用透视变换
            transformed = cv2.warpPerspective(
                mirrored, 
                perspective_matrix, 
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
                borderValue=(0, 0, 0, 0) if len(mirrored.shape) == 3 and mirrored.shape[2] == 4 else 0
            )
            
            # 创建掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 255)
            
            # 合并到结果图像
            if len(result.shape) == 3:
                for c in range(result.shape[2]):
                    result[:, :, c] = np.where(mask > 0, transformed[:, :, c], result[:, :, c])
            else:
                result = np.where(mask > 0, transformed, result)
            
            print("边缘镜像填充完成")
            
        except Exception as e:
            print(f"镜像填充出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 备用方案：使用简单的颜色填充
            try:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [points.astype(np.int32)], 255)
                
                # 使用灰色填充
                if len(result.shape) == 3:
                    result[mask > 0] = [128, 128, 128]
                else:
                    result[mask > 0] = 128
            except Exception as e2:
                print(f"备用填充也失败: {e2}")
        
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
        if len(area_points) != 4:
            return image
        
        result = image.copy()
        h, w = image.shape[:2]
        
        # 将点坐标转换为numpy数组
        points = np.array(area_points, dtype=np.float32)
        
        # 分析镜像区域的形状特征
        # 通常书脊/侧面是一个梯形或平行四边形
        left_edge_height = np.linalg.norm(points[3] - points[0])  # 左边缘高度
        right_edge_height = np.linalg.norm(points[2] - points[1])  # 右边缘高度
        top_edge_width = np.linalg.norm(points[1] - points[0])    # 上边缘宽度
        bottom_edge_width = np.linalg.norm(points[2] - points[3]) # 下边缘宽度
        
        # 判断是垂直的侧面还是水平的侧面
        is_vertical = (left_edge_height + right_edge_height) > (top_edge_width + bottom_edge_width)
        
        # 获取镜像区域的边界
        min_x = int(np.min(points[:, 0]))
        max_x = int(np.max(points[:, 0]))
        min_y = int(np.min(points[:, 1]))
        max_y = int(np.max(points[:, 1]))
        
        area_width = max_x - min_x
        area_height = max_y - min_y
        
        try:
            # 智能选择采样区域
            # 根据镜像区域的位置，决定从哪里采样
            center_x = np.mean(points[:, 0])
            
            # 判断采样区域
            if center_x < w * 0.3:  # 左侧镜像
                search_start = max_x + 10
                search_end = min(search_start + area_width * 3, w)
                
                if search_end > search_start and search_end <= w:
                    # 从主图案的左边缘采样，保持原始高度
                    sample_region = result[min_y:max_y, search_start:search_end].copy()
                    
                    # 保持原始宽高比，只取所需宽度
                    if sample_region.shape[1] > area_width:
                        sample_region = sample_region[:, :area_width]
                    
                    # 水平镜像，不调整大小
                    mirrored = cv2.flip(sample_region, 1)
                else:
                    mirrored = np.ones((area_height, area_width, result.shape[2]), dtype=result.dtype) * 200
            
            else:  # 右侧或其他位置
                search_end = min_x - 10
                search_start = max(0, search_end - area_width)  # 只取所需宽度
                
                if search_start < search_end:
                    sample_region = result[min_y:max_y, search_start:search_end].copy()
                    mirrored = cv2.flip(sample_region, 1)
                else:
                    mirrored = np.ones((area_height, area_width, result.shape[2]), dtype=result.dtype) * 200
            
            # 应用透视变换以适应梯形/平行四边形区域
            # 计算源图像的四个角点
            src_points = np.float32([
                [0, 0],
                [mirrored.shape[1], 0],
                [mirrored.shape[1], mirrored.shape[0]],
                [0, mirrored.shape[0]]
            ])
            
            # 计算目标区域的实际形状
            # 根据四个角点的相对位置计算变换
            dst_points = np.array([
                points[0],  # 左上
                points[1],  # 右上
                points[2],  # 右下
                points[3]   # 左下
            ], dtype=np.float32)
            
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # 应用透视变换
            warped = cv2.warpPerspective(
                mirrored, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # 创建掩码并合并
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 255)
            
            # 添加边缘羽化以获得更自然的过渡
            mask_blurred = cv2.GaussianBlur(mask, (7, 7), 0)  # 增加模糊半径从5到7
            alpha = mask_blurred.astype(float) / 255.0

            # 混合图像时使用更平滑的过渡
            if len(result.shape) == 3:
                for c in range(result.shape[2]):
                    result[:, :, c] = np.where(mask > 0, warped[:, :, c] * alpha + result[:, :, c] * (1 - alpha), result[:, :, c])
            else:
                result = np.where(mask > 0,  warped * alpha + result * (1 - alpha), result)
            
            result = result.astype(image.dtype)
            
        except Exception as e:
            print(f"智能镜像填充失败: {e}")
            # 使用基础镜像填充作为后备
            return ImageProcessor.mirror_fill_area(image, area_points, source_image)

        # return ImageProcessor.mirror_fill_area(image, area_points, source_image)
        return result

