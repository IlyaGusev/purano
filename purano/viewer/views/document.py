from flask import Blueprint, render_template

from purano.viewer.app import db
from purano.models import Document

bp = Blueprint('document', __name__)


@bp.route('/', methods=["GET"])
def list():
    documents = db.session.query(Document).all()
    return render_template("document/list.html", documents=documents)


@bp.route('/<int:document_id>', methods=["GET"])
def get(document_id):
    document = db.session.query(Document).get(document_id)
    ner = document.info.get_info().text_dp_ner
    return render_template("document/get.html", document=document, ner=ner)
