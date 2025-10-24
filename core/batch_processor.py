import os
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

from . import main_image_tool as mit
from .export_utils import export_multi_resolution
from .template_db import list_templates, init_db


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def batch_generate(design_paths: List[str],
                   template_records: Optional[List[Dict[str, Any]]] = None,
                   output_dir: str = 'outputs',
                   sensitivity: str = 'high',
                   edge_blur: int = 3,
                   show_corners: bool = False,
                   scales: List[float] = [1],
                   dpi: int = 300) -> Dict[str, List[str]]:
    """
    批量将设计图填充到模板(含黄/绿区域)并导出指定分辨率版本。
    为每个设计图创建独立的文件夹存放生成的图片。
    - design_paths: 设计图路径列表
    - template_records: 模板记录列表(来自模板DB)，若None则使用全部模板
    - output_dir: 输出目录
    - sensitivity, edge_blur, show_corners: 透视填充相关参数
    - scales: 导出倍数列表
    - dpi: 导出DPI
    返回按设计图分组的生成文件字典。
    """
    ensure_dir(output_dir)
    init_db()  # 保障DB存在
    if template_records is None:
        template_records = list_templates()

    generated_files_by_design: Dict[str, List[str]] = {}

    for design in design_paths:
        d_base = os.path.splitext(os.path.basename(design))[0]
        
        # 为每个设计图创建独立的文件夹
        design_output_dir = os.path.join(output_dir, d_base)
        ensure_dir(design_output_dir)
        
        generated_files_by_design[d_base] = []
        
        for tpl in template_records:
            tpl_path = tpl['path']
            tpl_name = tpl.get('name') or os.path.splitext(os.path.basename(tpl_path))[0]
            base_out = os.path.join(design_output_dir, f"{tpl_name}__{d_base}.png")

            # 调用现有算法进行透视填充
            try:
                result = mit.fill_yellow_region(
                    original_img_path=tpl_path,
                    fill_img_path=design,
                    output_path=base_out,
                    mode='perspective',
                    show_corners=show_corners,
                    anti_aliasing=True,
                    edge_blur=edge_blur,
                    sensitivity=sensitivity
                )
                
                # 检查是否成功生成文件
                if result is None or not os.path.exists(base_out):
                    print(f"跳过失败的处理: {tpl_name} + {d_base}")
                    continue
                
                # 读取输出并进行多倍导出（支持中文路径）
                try:
                    im = Image.open(base_out).convert('RGBA')
                except Exception as e:
                    print(f"无法读取生成的图片 {base_out}: {str(e)}")
                    continue
                im_np = np.array(im)
                export_multi_resolution(im_np, base_out, scales=scales, dpi=dpi, format='PNG')
                generated_files_by_design[d_base].append(base_out)
                
            except Exception as e:
                print(f"处理失败 {tpl_name} + {d_base}: {str(e)}")
                continue
            
    return generated_files_by_design