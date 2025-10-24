import os
from flask import Blueprint, request, render_template, current_app
from core.template_db import init_db, list_templates
from core.main_image_tool import fill_yellow_region

preview_bp = Blueprint('preview', __name__, url_prefix='/preview')

@preview_bp.route('/', methods=['GET', 'POST'])
def preview():
    init_db()
    templates = list_templates()
    
    if request.method == 'GET':
        return render_template('preview.html', templates=templates)
    
    # POST 处理
    design_file = request.files.get('design')
    template_id = request.form.get('template_id')
    sensitivity = float(request.form.get('sensitivity', 0.5))
    edge_blur = int(request.form.get('edge_blur', 5))
    show_corners = request.form.get('show_corners') == 'on'
    
    if not design_file or not template_id:
        return render_template('preview.html', templates=templates, error="请选择设计图和模板")
    
    # 找到对应的模板
    template_path = None
    for t in templates:
        if str(t['id']) == template_id:
            template_path = t['path']
            break
    
    if not template_path:
        return render_template('preview.html', templates=templates, error="模板不存在")
    
    # 保存上传的设计图
    app_dir = os.path.dirname(os.path.dirname(__file__))
    upload_dir = os.path.join(app_dir, 'static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    design_path = os.path.join(upload_dir, design_file.filename)
    design_file.save(design_path)
    
    # 处理图片
    try:
        output_dir = os.path.join(app_dir, 'static', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用与原始app.py相同的逻辑
        tpl_name = template_path.split('/')[-1].split('.')[0] if '/' in template_path else os.path.splitext(os.path.basename(template_path))[0]
        design_name = os.path.splitext(os.path.basename(design_path))[0]
        base_name = f"{tpl_name}__{design_name}.png"
        out_path = os.path.join(output_dir, base_name)
        
        # 转换sensitivity参数
        if isinstance(sensitivity, float):
            if sensitivity <= 0.3:
                sensitivity_str = 'low'
            elif sensitivity <= 0.7:
                sensitivity_str = 'medium'
            else:
                sensitivity_str = 'high'
        else:
            sensitivity_str = sensitivity
        
        fill_yellow_region(
            original_img_path=template_path,
            fill_img_path=design_path,
            output_path=out_path,
            mode='perspective',
            show_corners=show_corners,
            anti_aliasing=True,
            edge_blur=edge_blur,
            sensitivity=sensitivity_str
        )
        
        # 转换为相对于static的路径
        relative_path = f"outputs/{base_name}"
        return render_template('preview.html', templates=templates, 
                             output_url=f"/static/{relative_path}")
    
    except Exception as e:
        return render_template('preview.html', templates=templates, 
                             error=f"处理出错: {str(e)}")