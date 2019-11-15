from flask import Blueprint, render_template

bp = Blueprint('index', __name__)

@bp.route('/')
def hello_world():
    return render_template("index.html")
