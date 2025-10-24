import os
from typing import List, Tuple
import numpy as np
from PIL import Image


def _ensure_ext(base_path: str, format: str) -> str:
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = '.' + format.lower()
    return root + ext


def export_multi_resolution(img_rgba: np.ndarray,
                            base_path: str,
                            scales: List[float] = [1, 2, 3, 4],
                            dpi: int = 300,
                            format: str = 'PNG') -> List[str]:
    """
    以多个分辨率倍数导出图像，使用LANCZOS缩放，并写入DPI元数据。
    返回生成文件路径列表。
    """
    out_paths = []
    im = Image.fromarray(img_rgba, mode="RGBA")
    bw, bh = im.size
    root, ext = os.path.splitext(base_path)
    ext = '.' + format.lower()

    for s in scales:
        sw, sh = max(1, int(round(bw * s))), max(1, int(round(bh * s)))
        im_resized = im.resize((sw, sh), resample=Image.LANCZOS)
        out_path = f"{root}@{s}x{ext}"
        im_resized.save(out_path, dpi=(dpi, dpi))
        out_paths.append(out_path)
    return out_paths