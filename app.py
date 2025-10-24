import os
from flask import Flask

# 导入路由模块
from routes.main import main_bp
from routes.templates import templates_bp
from routes.preview import preview_bp
from routes.batch import batch_bp

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, 'data')
TPL_DIR = os.path.join(DATA_DIR, 'templates')
DESIGN_DIR = os.path.join(DATA_DIR, 'designs')
STATIC_DIR = os.path.join(APP_DIR, 'static')
OUTPUT_DIR = os.path.join(STATIC_DIR, 'outputs')

# 确保必要的目录存在
os.makedirs(TPL_DIR, exist_ok=True)
os.makedirs(DESIGN_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR)

# 注册蓝图
app.register_blueprint(main_bp)
app.register_blueprint(templates_bp)
app.register_blueprint(preview_bp)
app.register_blueprint(batch_bp)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)