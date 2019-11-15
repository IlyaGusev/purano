import os
import argparse
from flask import Flask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="Port", default="5001")
    parser.add_argument("--host", type=str, help="Host", default="0.0.0.0")
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    static_dir = os.path.abspath("purano/viewer/static")
    templates_dir = os.path.abspath("purano/viewer/templates")

    app = Flask(__name__, template_folder=templates_dir, static_folder=static_dir)
    app.config["SQLALCHEMY_DATABASE_URI"] = args.db_engine

    from purano.viewer.views.index import bp as index_bp
    app.register_blueprint(index_bp)

    app.run(host=args.host, port=args.port, use_reloader=False, debug=args.debug)
