import os
import cv2

from flask import Blueprint, request, jsonify

from config import Config
from models.database import db, User, Attendance
from face_engine.detector import decode_base64_frame, detect_faces
from face_engine.recognizer import train_model, predict_face


api = Blueprint("api", __name__)


def get_user_dataset_folder(user):
    safe_name = user.name.replace(" ", "_")

    folder_name = f"{user.id}_{safe_name}"

    return os.path.join(
        Config.DATASET_DIR,
        folder_name
    )


@api.route("/users", methods=["POST"])
def create_user():
    data = request.json

    user = User(
        name=data["name"],
        department=data.get("department")
    )

    db.session.add(user)
    db.session.commit()

    user_folder = get_user_dataset_folder(user)

    os.makedirs(
        user_folder,
        exist_ok=True
    )

    return jsonify({
        "message": "User berhasil dibuat",
        "user_id": user.id,
        "name": user.name,
        "dataset_folder": user_folder
    })

@api.route("/users", methods=["GET"])
def get_users():
    users = User.query.all()

    result = []

    for user in users:
        result.append({
            "id": user.id,
            "name": user.name,
            "department": user.department
        })

    return jsonify(result)

@api.route("/train", methods=["POST"])
def train():
    success, message = train_model()

    return jsonify({
        "success": success,
        "message": message
    })


@api.route("/register-face", methods=["POST"])
def register_face():
    data = request.json

    user_id = data["user_id"]
    image = data["image"]

    user = User.query.get(user_id)

    if user is None:
        return jsonify({
            "success": False,
            "message": "User tidak ditemukan"
        })

    frame = decode_base64_frame(image)
    faces, gray = detect_faces(frame)

    if len(faces) == 0:
        return jsonify({
            "success": False,
            "message": "Tidak ada wajah"
        })

    x, y, w, h = faces[0]

    face_color = frame[y:y+h, x:x+w]

    face_color = cv2.resize(
        face_color,
        Config.FACE_SIZE
    )

    user_folder = get_user_dataset_folder(user)

    os.makedirs(
        user_folder,
        exist_ok=True
    )

    count = len(os.listdir(user_folder)) + 1

    filename = os.path.join(
        user_folder,
        f"{count}.jpg"
    )

    cv2.imwrite(
        filename,
        face_color,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    return jsonify({
        "success": True,
        "user_id": user.id,
        "name": user.name,
        "file": filename
    })


@api.route("/recognize", methods=["POST"])
def recognize():
    data = request.json
    image = data["image"]

    frame = decode_base64_frame(image)
    faces, gray = detect_faces(frame)

    if len(faces) == 0:
        return jsonify({
            "status": "no_face",
            "message": "Tidak ada wajah"
        })

    x, y, w, h = faces[0]

    face_gray = gray[y:y+h, x:x+w]

    user_id, confidence, status = predict_face(face_gray)

    if status == "recognized":
        user = User.query.get(user_id)

        if user is None:
            return jsonify({
                "status": "unknown",
                "message": "User tidak ditemukan di database",
                "confidence": confidence
            })

        attendance = Attendance(user_id=user_id)
        db.session.add(attendance)
        db.session.commit()

        return jsonify({
            "status": "recognized",
            "user_id": user.id,
            "name": user.name,
            "confidence": confidence
        })

    return jsonify({
        "status": "unknown",
        "confidence": confidence
    })