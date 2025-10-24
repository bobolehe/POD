from typing import Tuple
import numpy as np
from PIL import Image, ImageFilter


def add_drop_shadow(fg_rgba: np.ndarray,
                    offset: Tuple[int, int] = (10, 10),
                    blur_radius: int = 12,
                    color: Tuple[int, int, int, int] = (0, 0, 0, 255),
                    opacity: float = 0.6) -> np.ndarray:
    """
    为前景图添加投影(软阴影)，返回叠加阴影后的RGBA。
    - offset: 阴影相对前景偏移
    - blur_radius: 阴影模糊程度
    - color: 阴影颜色(含Alpha)
    - opacity: 阴影不透明度(0-1)
    """
    fg = Image.fromarray(fg_rgba, mode="RGBA")
    w, h = fg.size

    # 构造阴影图: 使用前景Alpha作为形状掩码
    alpha = fg.split()[3]
    # 阴影颜色+Alpha(按opacity缩放)
    shadow_alpha = int(max(0, min(255, color[3] * opacity)))
    shadow_base = Image.new("RGBA", (w, h), (color[0], color[1], color[2], shadow_alpha))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow.paste(shadow_base, (0, 0), alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 组合到画布
    canvas = Image.new("RGBA", (w + abs(offset[0]), h + abs(offset[1])), (0, 0, 0, 0))
    ox = max(0, offset[0])
    oy = max(0, offset[1])
    canvas.paste(shadow, (ox, oy), shadow)
    canvas.paste(fg, (0, 0), fg)
    return np.array(canvas)


def apply_gaussian_blur_rgba(img_rgba: np.ndarray, radius: int = 3) -> np.ndarray:
    """对RGBA图像整体进行高斯模糊"""
    im = Image.fromarray(img_rgba, mode="RGBA")
    out = im.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(out)


def to_grayscale_preserve_alpha(img_rgba: np.ndarray) -> np.ndarray:
    """转换为灰度并保留Alpha通道"""
    im = Image.fromarray(img_rgba, mode="RGBA")
    # 分离通道
    r, g, b, a = im.split()
    # 加权灰度: 0.299R + 0.587G + 0.114B
    rgb = Image.merge("RGB", (r, g, b))
    gray = rgb.convert("L")
    out = Image.merge("RGBA", (gray, gray, gray, a))
    return np.array(out)