import argparse
import os
from typing import List

from .template_db import init_db, add_template, list_templates, delete_template
from .batch_processor import batch_generate
from .image_pipeline import load_image_rgba, rotate_rgba, save_image_rgba
from .effects import add_drop_shadow, apply_gaussian_blur_rgba, to_grayscale_preserve_alpha


def cmd_add_template(args):
    init_db()
    tid = add_template(name=args.name, path=args.path)
    print(f"添加模板成功: id={tid}, name={args.name}, path={args.path}")


def cmd_list_templates(args):
    init_db()
    templates = list_templates()
    if not templates:
        print("暂无模板记录")
        return
    for t in templates:
        print(f"id={t['id']} name={t['name']} path={t['path']} size={t['width']}x{t['height']}")


def cmd_delete_template(args):
    if delete_template(args.id):
        print(f"删除模板成功: id={args.id}")
    else:
        print(f"删除模板失败或不存在: id={args.id}")


def _collect_files(dir_path: str) -> List[str]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    out = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
    return out


def cmd_batch_generate(args):
    init_db()
    if args.design_dir:
        design_paths = _collect_files(args.design_dir)
    else:
        design_paths = args.design
    templates = list_templates()
    scales = [float(s) for s in args.scales.split(',')] if args.scales else [1]

    files = batch_generate(
        design_paths=design_paths,
        template_records=templates,
        output_dir=args.output_dir,
        sensitivity=args.sensitivity,
        edge_blur=args.edge_blur,
        show_corners=args.show_corners,
        scales=scales,
        dpi=args.dpi
    )
    print(f"生成完成，共 {len(files)} 个基础文件")


def cmd_rotate(args):
    img = load_image_rgba(args.input)
    out = rotate_rgba(img, args.angle)
    save_image_rgba(args.output, out, dpi=(args.dpi, args.dpi) if args.dpi else None)
    print(f"旋转完成: {args.input} -> {args.output}, angle={args.angle}")


def cmd_shadow(args):
    img = load_image_rgba(args.input)
    out = add_drop_shadow(img, offset=(args.offset_x, args.offset_y), blur_radius=args.blur, opacity=args.opacity)
    save_image_rgba(args.output, out, dpi=(args.dpi, args.dpi) if args.dpi else None)
    print(f"阴影完成: {args.input} -> {args.output}")


def cmd_blur(args):
    img = load_image_rgba(args.input)
    out = apply_gaussian_blur_rgba(img, radius=args.radius)
    save_image_rgba(args.output, out, dpi=(args.dpi, args.dpi) if args.dpi else None)
    print(f"模糊完成: {args.input} -> {args.output}")


def cmd_grayscale(args):
    img = load_image_rgba(args.input)
    out = to_grayscale_preserve_alpha(img)
    save_image_rgba(args.output, out, dpi=(args.dpi, args.dpi) if args.dpi else None)
    print(f"灰度完成: {args.input} -> {args.output}")


def build_parser():
    p = argparse.ArgumentParser(description='POD工具命令行')
    sub = p.add_subparsers()

    # 模板
    sp = sub.add_parser('add-template', help='添加模板到数据库')
    sp.add_argument('--name', required=True)
    sp.add_argument('--path', required=True)
    sp.set_defaults(func=cmd_add_template)

    sp = sub.add_parser('list-templates', help='列出模板')
    sp.set_defaults(func=cmd_list_templates)

    sp = sub.add_parser('delete-template', help='删除模板')
    sp.add_argument('--id', type=int, required=True)
    sp.set_defaults(func=cmd_delete_template)

    # 批量
    sp = sub.add_parser('batch-generate', help='批量生成预览')
    sp.add_argument('--design-dir', help='设计图目录(递归)')
    sp.add_argument('--design', nargs='*', help='指定设计图路径列表')
    sp.add_argument('--output-dir', default='outputs')
    sp.add_argument('--sensitivity', default='high', choices=['low', 'medium', 'high'])
    sp.add_argument('--edge-blur', type=int, default=3)
    sp.add_argument('--show-corners', action='store_true')
    sp.add_argument('--scales', default='1,2,3,4', help='导出倍数列表, 逗号分隔')
    sp.add_argument('--dpi', type=int, default=300)
    sp.set_defaults(func=cmd_batch_generate)

    # 基础图像操作
    sp = sub.add_parser('rotate-image', help='旋转图像(90/180/270)')
    sp.add_argument('--input', required=True)
    sp.add_argument('--output', required=True)
    sp.add_argument('--angle', type=int, choices=[0, 90, 180, 270], required=True)
    sp.add_argument('--dpi', type=int)
    sp.set_defaults(func=cmd_rotate)

    sp = sub.add_parser('shadow', help='添加阴影')
    sp.add_argument('--input', required=True)
    sp.add_argument('--output', required=True)
    sp.add_argument('--offset-x', type=int, default=10)
    sp.add_argument('--offset-y', type=int, default=10)
    sp.add_argument('--blur', type=int, default=12)
    sp.add_argument('--opacity', type=float, default=0.6)
    sp.add_argument('--dpi', type=int)
    sp.set_defaults(func=cmd_shadow)

    sp = sub.add_parser('blur', help='高斯模糊')
    sp.add_argument('--input', required=True)
    sp.add_argument('--output', required=True)
    sp.add_argument('--radius', type=int, default=3)
    sp.add_argument('--dpi', type=int)
    sp.set_defaults(func=cmd_blur)

    sp = sub.add_parser('grayscale', help='黑白转换(保留Alpha)')
    sp.add_argument('--input', required=True)
    sp.add_argument('--output', required=True)
    sp.add_argument('--dpi', type=int)
    sp.set_defaults(func=cmd_grayscale)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()