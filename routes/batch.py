import os
from flask import Blueprint, request, render_template, current_app
from core.template_db import init_db, list_templates
from core.batch_processor import batch_generate

batch_bp = Blueprint('batch', __name__, url_prefix='/batch')

@batch_bp.route('/', methods=['GET', 'POST'])
def batch():
    init_db()
    templates = list_templates()
    
    if request.method == 'GET':
        return render_template('batch.html', templates=templates)
    
    # POST 处理 - 使用与原始app.py相同的逻辑
    tpl_ids = request.form.getlist('template_ids')
    tpl_ids = [int(x) for x in tpl_ids]
    sensitivity = request.form.get('sensitivity', 'high')
    edge_blur = int(request.form.get('edge_blur', '3'))
    show_corners = bool(request.form.get('show_corners'))
    scales_raw = request.form.get('scales', '1,2,3,4')
    dpi = int(request.form.get('dpi', '300'))
    scales = []
    for s in scales_raw.split(','):
        s = s.strip()
        if not s:
            continue
        try:
            scales.append(float(s))
        except:
            pass
    if not scales:
        scales = [1]

    files = request.files.getlist('files')
    design_paths = []
    app_dir = os.path.dirname(os.path.dirname(__file__))
    design_dir = os.path.join(app_dir, 'data', 'designs')
    for f in files:
        if not f.filename:
            continue
        save_path = os.path.join(design_dir, f.filename)
        f.save(save_path)
        design_paths.append(save_path)

    selected_templates = [r for r in templates if r['id'] in tpl_ids]
    from pathlib import Path
    output_dir = os.path.join(app_dir, 'static', 'images', 'outputs')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        generated_files_by_design = batch_generate(
            design_paths=design_paths,
            template_records=selected_templates,
            output_dir=output_dir,
            sensitivity=sensitivity,
            edge_blur=edge_blur,
            show_corners=show_corners,
            scales=scales,
            dpi=dpi
        )

        # 为每个设计图生成对应的输出URL
        output_urls = []
        for design_name, file_paths in generated_files_by_design.items():
            for file_path in file_paths:
                # 获取相对于static目录的路径
                relative_path = os.path.relpath(file_path, os.path.join(app_dir, 'static'))
                output_urls.append(f"/static/{relative_path.replace(os.sep, '/')}")
        
        return render_template('batch.html', templates=templates, output_urls=output_urls)
    
    except Exception as e:
        return render_template('batch.html', templates=templates, 
                             error=f"处理出错: {str(e)}")