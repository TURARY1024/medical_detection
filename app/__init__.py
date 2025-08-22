from flask import Flask, send_from_directory
from app.route import index, upload_image, match_drug  # ✅ 確保有匯入 match_drug
import os

def create_app():
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = os.path.abspath("uploads")

    # 註冊路由
    app.add_url_rule("/", "/", index)
    app.add_url_rule("/upload", "upload", upload_image, methods=["POST"])
    app.add_url_rule("/match", "match", match_drug, methods=["POST"])  # ✅ 新增這行

    @app.route("/uploads/<filename>", methods=["GET"])
    def uploaded_file(filename):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        print("Trying to access file at:", file_path)
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    return app
