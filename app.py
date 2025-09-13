from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import chatbotai
import os

app = Flask(__name__)
CORS(app)

# Route để load giao diện index.html (nằm cùng thư mục với app.py)
@app.route("/")
def home():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()
    if not user_input:
        return jsonify({"reply": "Bạn chưa nhập câu hỏi."})
    
    match = chatbotai.find_best_match(user_input)
    if match:
        reply = chatbotai.data[match]
    else:
        reply = "Mình chưa biết câu trả lời. Bạn có thể dạy mình!"
    return jsonify({"reply": reply})


@app.route("/train", methods=["POST"])
def train():
    user_input = request.json.get("question", "").strip().lower()
    answer = request.json.get("answer", "").strip()
    if user_input and answer:
        chatbotai.data[user_input] = answer
        chatbotai.save_data()
        return jsonify({"status": "ok"})
    return jsonify({"status": "fail"})

@app.route("/upload", methods=["POST"])
def upload_excel():
    if "file" not in request.files:
        return jsonify({"error": "❌ Không có file nào được gửi!"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".xlsx"):
        return jsonify({"error": "❌ File không hợp lệ, chỉ chấp nhận .xlsx"}), 400

    try:
        file_path = os.path.join(chatbotai.UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Train luôn với dữ liệu Excel
        added = chatbotai.train_from_excel(file_path)

        return jsonify({
            "message": f"✅ Upload & train thành công: {added} câu mới từ {file.filename}"
        })
    except Exception as e:
        return jsonify({"error": f"❌ Lỗi xử lý file: {str(e)}"}), 500

@app.route("/train_excel", methods=["POST"])
def train_excel():
    chatbotai.train_from_excel(chatbotai.EXCEL_FILE)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
