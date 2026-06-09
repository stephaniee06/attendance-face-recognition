import os
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from routes.api import api

from config import Config
from models.database import db

app = Flask(__name__)
app.config.from_object(Config)

CORS(app)

db.init_app(app)
app.register_blueprint(api, url_prefix="/api")

with app.app_context():
    db.create_all()

# --- FUNGSI UTK MEMBACA FOLDER DATASET LOKAL ---
def get_dataset_users():
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    registered_users = []

    if os.path.exists(dataset_path):
        # Membaca folder: 1_Stephanie_Halim, 2_Aulia_Aca_Azzahra, dst.
        folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        folders.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 999)

        for folder in folders:
            parts = folder.split('_')
            # Ambil nomor ID depan folder
            user_id = int(parts[0]) if parts[0].isdigit() else folder
            # Buat nama bersih untuk dropdown (contoh: "Stephanie Halim (10 faces)")
            display_name = " ".join(parts[1:]) if len(parts) > 1 else folder
            
            folder_path = os.path.join(dataset_path, folder)
            face_count = len([img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])

            # Struktur dikembalikan dalam format dinamis "Nama (X faces)" sesuai keinginan kamu di UI
            registered_users.append({
                'id': user_id,
                'name': f"{display_name} ({face_count} faces)",
                'face_count': face_count
            })
    return registered_users

@app.route("/")
def home():
    return "Backend Face Attendance System is running. Access the camera interface at /camera"

# --- MODIFIKASI ROUTE /camera AGAR OTOMATIS SWAP BASE_URL KE LOKAL ---
@app.route("/camera")
def camera():
    response = render_template("camera.html")
    # Trik override: Mengganti URL Render di HTML menjadi URL lokal secara realtime saat diakses
    local_url = "http://127.0.0.1:5000"
    render_url = "https://attendance-face-recognition-3yy6.com"
    render_url_alt = "https://attendance-face-recognition-3yy6.onrender.com"
    
    response = response.replace(render_url_alt, local_url)
    response = response.replace(render_url, local_url)
    return response

# --- ENDPOINT API UTK MENYUPLAI DATA JAVASCRIPT ---
@app.route("/api/users", methods=["GET"])
def get_users_api_override():
    users = get_dataset_users()
    return jsonify(users)

if __name__ == "__main__":
    app.run(debug=True)