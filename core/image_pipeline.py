import os
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageFilter

# RGBA工具函数

def load_image_rgba(path: str) -> np.ndarray:
    """
    使用Pillow读取图像并统一转换为RGBA，返回numpy数组(H, W, 4)，uint8。
    """
    img = Image.open(path).convert("RGBA")
    return np.array(img)


def save_image_rgba(path: str, img_rgba: np.ndarray, dpi: Optional[Tuple[int, int]] = None) -> None:
    """
    保存RGBA图像到指定路径，支持写入DPI元数据。
    """
    im = Image.fromarray(img_rgba, mode="RGBA")
    if dpi:
        im.save(path, dpi=dpi)
    else:
        im.save(path)


def resize_rgba(img_rgba: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """使用Pillow的LANCZOS进行高质量重采样到指定尺寸(size=(W,H))"""
    im = Image.fromarray(img_rgba, mode="RGBA")
    im_resized = im.resize(size, resample=Image.LANCZOS)
    return np.array(im_resized)


def rotate_rgba(img_rgba: np.ndarray, angle: int) -> np.ndarray:
    """
    旋转图像，支持0/90/180/270度，保持RGBA。
    """
    angle = angle % 360
    im = Image.fromarray(img_rgba, mode="RGBA")
    if angle == 0:
        return img_rgba
    elif angle == 90:
        out = im.transpose(Image.ROTATE_90)
    elif angle == 180:
        out = im.transpose(Image.ROTATE_180)
    elif angle == 270:
        out = im.transpose(Image.ROTATE_270)
    else:
        # 其他角度使用双线性以避免过度模糊
        out = im.rotate(angle, resample=Image.BILINEAR, expand=True)
    return np.array(out)


def alpha_composite(bg_rgba: np.ndarray, fg_rgba: np.ndarray, x: int = 0, y: int = 0) -> np.ndarray:
    """
    将前景fg按位置(x,y)叠加到背景bg，使用Alpha进行合成，返回新的RGBA。
    若前景越界，自动裁切可视区域。
    """
    bg = Image.fromarray(bg_rgba, mode="RGBA")
    fg = Image.fromarray(fg_rgba, mode="RGBA")

    # 处理越界裁切
    bw, bh = bg.size
    fw, fh = fg.size
    if x >= bw or y >= bh:
        return bg_rgba.copy()
    crop_left = 0 if x >= 0 else -x
    crop_top = 0 if y >= 0 else -y
    crop_right = fw if x + fw <= bw else bw - x
    crop_bottom = fh if y + fh <= bh else bh - y
    if crop_right <= 0 or crop_bottom <= 0:
        return bg_rgba.copy()

    fg_cropped = fg.crop((crop_left, crop_top, crop_right, crop_bottom))
    paste_x = max(0, x)
    paste_y = max(0, y)

    bg.paste(fg_cropped, (paste_x, paste_y), fg_cropped)
    return np.array(bg)


def apply_mask_to_alpha(img_rgba: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    """
    将灰度掩码(0-255)应用到图像的Alpha通道，返回新的RGBA。
    """
    h, w = img_rgba.shape[:2]
    mask = mask_gray
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != (h, w):
        raise ValueError("mask尺寸与图像不匹配")
    out = img_rgba.copy()
    # 按比例缩放alpha
    alpha = out[..., 3].astype(np.float32) / 255.0
    mask_f = mask.astype(np.float32) / 255.0
    new_alpha = np.clip(alpha * mask_f, 0.0, 1.0)
    out[..., 3] = (new_alpha * 255.0).astype(np.uint8)
    return out


def cv_bgr_to_rgba(bgr_img: np.ndarray) -> np.ndarray:
    """将OpenCV的BGR图像转换为RGBA(Alpha=255)"""
    import cv2
    rgba = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGBA)
    return rgba


def rgba_to_cv_bgr(rgba_img: np.ndarray) -> np.ndarray:
    """将RGBA图像转换为OpenCV的BGR格式(丢弃Alpha)"""
    import cv2
    bgr = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)
    return bgr