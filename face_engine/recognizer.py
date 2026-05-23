import os
import cv2
import numpy as np

from config import Config
from face_engine.detector import detect_faces


recognizer = cv2.face.LBPHFaceRecognizer_create()


def prepare_training_face(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    faces, gray = detect_faces(image)

    if len(faces) > 0:
        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        face = gray[y:y+h, x:x+w]
    else:
        face = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

    if face is None:
        return None

    return cv2.resize(
        face,
        Config.FACE_SIZE
    )


def train_model():
    faces = []
    labels = []
    skipped = 0

    for folder_name in os.listdir(Config.DATASET_DIR):
        user_folder = os.path.join(
            Config.DATASET_DIR,
            folder_name
        )

        if not os.path.isdir(user_folder):
            continue

        try:
            user_id = int(
                folder_name.split("_")[0]
            )
        except ValueError:
            continue

        for filename in os.listdir(user_folder):
            image_path = os.path.join(
                user_folder,
                filename
            )

            image = prepare_training_face(image_path)

            if image is None:
                skipped += 1
                continue

            faces.append(image)
            labels.append(user_id)

    if len(faces) == 0:
        return False, "Dataset masih kosong"

    recognizer.train(
        faces,
        np.array(labels)
    )

    os.makedirs(
        Config.MODEL_DIR,
        exist_ok=True
    )

    recognizer.save(
        Config.MODEL_PATH
    )

    return True, (
        f"Model berhasil dilatih dengan {len(faces)} foto"
        f" ({skipped} foto dilewati)"
    )


def predict_face(gray_face):
    if not os.path.exists(Config.MODEL_PATH):
        return None, None, "Model belum dilatih"

    recognizer.read(
        Config.MODEL_PATH
    )

    face = cv2.resize(
        gray_face,
        Config.FACE_SIZE
    )

    label, confidence = recognizer.predict(face)

    if confidence < Config.CONFIDENCE_THRESHOLD:
        return int(label), round(confidence, 2), "recognized"

    return None, round(confidence, 2), "unknown"
