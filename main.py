from flask import Flask, request, jsonify
from pyngrok import ngrok
import torch
import whisper
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 檢查 GPU 可用性，使用最快的設備
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=DEVICE)

@app.route("/<name>")
def home(name):
    return f"<h1>hello {name}</h1>"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    try:
        result = model.transcribe(filepath)
        os.remove(filepath)  # 處理完後刪除檔案
        return jsonify({"text": result["text"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # public_url = ngrok.connect(5000).public_url
    # print("請使用此連結:", public_url + "/transcribe")
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(port=5000)