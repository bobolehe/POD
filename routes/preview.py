import os
import tempfile
import time
import threading
import shutil
from flask import Blueprint, request, render_template, send_file
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
    design_file = request.files.get('file')
    template_id = request.form.get('template_id')
    sensitivity_raw = request.form.get('sensitivity', '0.5')
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
    
    # 初始化变量
    design_path = None
    out_path = None
    
    try:
        # 创建临时文件用于保存上传的设计文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_design:
            design_file.save(temp_design.name)
            design_path = temp_design.name
        
        # 创建临时文件用于输出
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_output:
            out_path = temp_output.name
        # 转换sensitivity参数
        try:
            # 尝试转换为float（兼容数值输入）
            sensitivity_float = float(sensitivity_raw)
            if sensitivity_float <= 0.3:
                sensitivity_str = 'low'
            elif sensitivity_float <= 0.7:
                sensitivity_str = 'medium'
            else:
                sensitivity_str = 'high'
        except ValueError:
            # 如果无法转换为float，直接使用字符串值
            sensitivity_str = sensitivity_raw if sensitivity_raw in ['low', 'medium', 'high'] else 'medium'
        
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
        
        # 将临时文件复制到static目录用于显示（但会定时删除）
        timestamp = str(int(time.time()))
        output_filename = f"preview_{timestamp}.png"
        app_dir = os.path.dirname(os.path.dirname(__file__))
        static_output_dir = os.path.join(app_dir, 'static', 'images', 'outputs')
        os.makedirs(static_output_dir, exist_ok=True)
        static_output_path = os.path.join(static_output_dir, output_filename)
        
        # 复制临时文件到static目录
        shutil.copy2(out_path, static_output_path)
        
        # 设置定时删除（30秒后删除预览文件）
        def cleanup_preview_file():
            time.sleep(30)
            try:
                if os.path.exists(static_output_path):
                    os.remove(static_output_path)
            except:
                pass
        
        cleanup_thread = threading.Thread(target=cleanup_preview_file)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        # 返回结果页面，显示生成的图像
        output_url = f"/static/images/outputs/{output_filename}"
        return render_template('preview.html', templates=templates, 
                             output_url=output_url)
    
    except Exception as e:
        return render_template('preview.html', templates=templates, 
                             error=f"处理出错: {str(e)}")
    
    finally:
        # 清理临时文件
        try:
            if os.path.exists(design_path):
                os.remove(design_path)
            if os.path.exists(out_path):
                os.remove(out_path)
        except:
            pass