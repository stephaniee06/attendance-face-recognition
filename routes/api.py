import os
import shutil
import cv2

from flask import Blueprint, request, jsonify

from config import Config
from models.database import db, User, Attendance
from face_engine.detector import decode_base64_frame, detect_faces
from face_engine.recognizer import train_model, update_model, predict_face


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
        "message": "User created successfully",
        "user_id": user.id,
        "name": user.name,
        "dataset_folder": user_folder
    })

@api.route("/users", methods=["GET"])
def get_users():
    users = User.query.all()

    result = []

    for user in users:
        user_folder = get_user_dataset_folder(user)
        face_count = 0
        if os.path.exists(user_folder):
            face_count = len([
                f for f in os.listdir(user_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

        result.append({
            "id": user.id,
            "name": user.name,
            "department": user.department,
            "face_count": face_count
        })

    return jsonify(result)


@api.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    user = User.query.get(user_id)

    if user is None:
        return jsonify({
            "success": False,
            "message": "User not found"
        }), 404

    deleted_attendance = Attendance.query.filter_by(user_id=user_id).delete()

    # Delete dataset folder
    user_folder = get_user_dataset_folder(user)
    folder_deleted = False
    if os.path.exists(user_folder):
        shutil.rmtree(user_folder)
        folder_deleted = True

    user_name = user.name

    db.session.delete(user)
    db.session.commit()

    print(f"[API delete] Deleted user '{user_name}' (ID: {user_id}), "
          f"{deleted_attendance} attendance records, "
          f"folder deleted: {folder_deleted}")

    return jsonify({
        "success": True,
        "message": f"User '{user_name}' deleted successfully. Please retrain the model.",
        "deleted_attendance": deleted_attendance,
        "folder_deleted": folder_deleted,
        "retrain_needed": True
    })


@api.route("/train", methods=["POST"])
def train():
    valid_user_ids = [u.id for u in User.query.all()]
    print(f"[API train] Starting training with {len(valid_user_ids)} registered users")

    success, message = train_model(valid_user_ids=valid_user_ids)

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
            "message": "User not found"
        })

    frame = decode_base64_frame(image)
    faces, gray = detect_faces(frame)

    if len(faces) == 0:
        return jsonify({
            "success": False,
            "message": "No face detected"
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

    face_count = len([
        f for f in os.listdir(user_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    face_gray = gray[y:y+h, x:x+w]
    model_updated, update_msg = update_model(face_gray, user.id)
    print(f"[API register-face] {update_msg}")

    return jsonify({
        "success": True,
        "user_id": user.id,
        "name": user.name,
        "face_count": face_count,
        "file": filename,
        "model_updated": model_updated,
        "update_message": update_msg
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
            "message": "No face detected"
        })

    x, y, w, h = faces[0]

    face_gray = gray[y:y+h, x:x+w]

    user_id, confidence, status = predict_face(face_gray)

    if status == "recognized":
        user = User.query.get(user_id)

        if user is None:
            print(f"[API recognize] Face predicted as ID {user_id}, but ID not found in database.")
            return jsonify({
                "status": "unknown",
                "message": "User not found in database",
                "confidence": confidence
            })

        attendance = Attendance(user_id=user_id)
        db.session.add(attendance)
        db.session.commit()

        print(f"[API recognize] Recognized: {user.name} (ID: {user.id}) | Confidence: {confidence}")
        return jsonify({
            "status": "recognized",
            "user_id": user.id,
            "name": user.name,
            "confidence": confidence
        })

    print(f"[API recognize] Unknown face | Distance/Confidence: {confidence}")
    return jsonify({
        "status": "unknown",
        "confidence": confidence
    })