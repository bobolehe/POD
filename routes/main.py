from flask import Blueprint, render_template
from core.template_db import init_db

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    init_db()
    return render_template('index.html')