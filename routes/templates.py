import os
from flask import Blueprint, request, redirect, url_for, render_template, current_app
from core.template_db import init_db, add_template, list_templates, delete_template

templates_bp = Blueprint('templates', __name__, url_prefix='/templates')

@templates_bp.route('/', methods=['GET'])
def templates_list():
    init_db()
    records = list_templates()
    class R: pass
    rs = []
    for r in records:
        o = R(); o.id = r['id']; o.name = r['name']; o.path = r['path']
        rs.append(o)
    return render_template('templates.html', templates=rs)

@templates_bp.route('/add', methods=['POST'])
def add_template_route():
    init_db()
    name = request.form.get('name', '').strip()
    f = request.files.get('file')
    if not f or not name:
        return redirect(url_for('templates.templates_list'))
    
    # 获取模板目录路径
    app_dir = os.path.dirname(os.path.dirname(__file__))
    tpl_dir = os.path.join(app_dir, 'data', 'templates')
    os.makedirs(tpl_dir, exist_ok=True)
    
    filename = f.filename
    save_path = os.path.join(tpl_dir, filename)
    f.save(save_path)
    add_template(name=name, path=save_path)
    return redirect(url_for('templates.templates_list'))

@templates_bp.route('/delete/<int:tid>', methods=['POST'])
def delete_template_route(tid: int):
    init_db()
    delete_template(tid)
    return redirect(url_for('templates.templates_list'))