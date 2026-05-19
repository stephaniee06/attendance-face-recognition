import os
import cv2
import numpy as np

from config import Config


recognizer = cv2.face.LBPHFaceRecognizer_create()


def train_model():
    faces = []
    labels = []

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

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = cv2.resize(
                image,
                Config.FACE_SIZE
            )

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

    return True, "Model berhasil dilatih"


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