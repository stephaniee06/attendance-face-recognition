import os
import cv2
import numpy as np

from config import Config
from face_engine.detector import detect_faces

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def align_face(gray_face):
    eyes = eye_cascade.detectMultiScale(
        gray_face,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    if len(eyes) < 2:
        return gray_face

    eyes = sorted(eyes, key=lambda e: e[0])
    left_eye = eyes[0]
    right_eye = eyes[1]

    left_center = (
        left_eye[0] + left_eye[2] // 2,
        left_eye[1] + left_eye[3] // 2
    )
    right_center = (
        right_eye[0] + right_eye[2] // 2,
        right_eye[1] + right_eye[3] // 2
    )

    dy = right_center[1] - left_center[1]
    dx = right_center[0] - left_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    if abs(angle) < 1.0 or abs(angle) > 30.0:
        return gray_face

    h, w = gray_face.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(
        gray_face, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return aligned


def preprocess_face(gray_face):
    resized = cv2.resize(gray_face, Config.FACE_SIZE)
    equalized = clahe.apply(resized)
    return equalized


def augment_face(face_image):
    augmented = []

    # Horizontal flip
    augmented.append(cv2.flip(face_image, 1))

    # Brighter
    bright = cv2.convertScaleAbs(face_image, alpha=1.15, beta=15)
    augmented.append(bright)

    # Darker
    dark = cv2.convertScaleAbs(face_image, alpha=0.85, beta=-15)
    augmented.append(dark)

    return augmented


def prepare_training_face(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]

    if max(w, h) <= 400:
        return preprocess_face(gray)
    
    faces, gray = detect_faces(image)

    if len(faces) > 0:
        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        face = gray[y:y+h, x:x+w]
        return preprocess_face(face)
    else:
        return None


def train_model(valid_user_ids=None):
    print(f"Training model with the following configuration:")
    print(f"Dataset directory: {Config.DATASET_DIR}")
    print(f"LBPH params: radius=2, neighbors=16, grid=10x10")
    print(f"Preprocessing: CLAHE + Eye Alignment")
    print(f"Augmentation: flip + brightness variations")
    if valid_user_ids is not None:
        print(f"Valid user IDs from DB: {valid_user_ids}")

    faces = []
    labels = []
    skipped = 0

    if not os.path.exists(Config.DATASET_DIR):
        print(f"[Error] Dataset directory not found!")
        return False, "Dataset directory not found"

    folders = os.listdir(Config.DATASET_DIR)
    print(f"Found {len(folders)} item(s) in dataset directory.")

    for folder_name in folders:
        user_folder = os.path.join(Config.DATASET_DIR, folder_name)
        if not os.path.isdir(user_folder):
            continue

        try:
            user_id = int(folder_name.split("_")[0])
        except ValueError:
            print(f"[Skip] '{folder_name}' passed - folder name should start with user ID (e.g., '1_JohnDoe')")
            continue
        
        if valid_user_ids is not None and user_id not in valid_user_ids:
            print(f"[Skip] '{folder_name}' - User ID {user_id} not found in database (orphan data)")
            continue

        print(f"\nProcessing Folder: '{folder_name}' (User ID: {user_id})")
        user_faces_count = 0
        user_augmented_count = 0
        user_skipped_count = 0

        for filename in os.listdir(user_folder):
            image_path = os.path.join(user_folder, filename)
            image = prepare_training_face(image_path)

            if image is None:
                user_skipped_count += 1
                skipped += 1
                print(f"[Failed/Skip] {filename}: Face not detected or image unreadable")
                continue

            faces.append(image)
            labels.append(user_id)
            user_faces_count += 1

            for aug_face in augment_face(image):
                faces.append(aug_face)
                labels.append(user_id)
                user_augmented_count += 1

            print(f"[Success] {filename}: Face + {len(augment_face(image))} augmentations")

        total_from_user = user_faces_count + user_augmented_count
        print(f"Result from '{folder_name}': {user_faces_count} original + {user_augmented_count} augmentations = {total_from_user} total ({user_skipped_count} skipped)")

    print(f"\nSummary:")
    print(f"Total training faces: {len(faces)} (augmented included)")
    print(f"Total skipped: {skipped}")

    if len(faces) == 0:
        print("[Error] No training faces found!")
        return False, "Dataset is empty or no valid faces detected"

    print("\nTraining model...")
    recognizer.train(faces, np.array(labels))

    print(f"Model saved to: {Config.MODEL_PATH}")
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    recognizer.save(Config.MODEL_PATH)
    print("Training Completed & Model Successfully Saved\n")

    return True, (
        f"Model trained with {len(faces)} photos"
        f" ({skipped} photos skipped)"
    )


def update_model(gray_face, user_id):
    if not os.path.exists(Config.MODEL_PATH):
        print("[Update] No existing model found, cannot update. Run full training first.")
        return False, "Model not trained. Please train the model first."

    face = preprocess_face(gray_face)

    faces = [face] + augment_face(face)
    labels = np.array([user_id] * len(faces))

    recognizer.update(faces, labels)
    recognizer.save(Config.MODEL_PATH)

    print(f"[Update] Model updated with {len(faces)} faces (1 original + {len(faces)-1} augmented) for user ID {user_id}")
    return True, f"Model updated with {len(faces)} faces for user ID {user_id}"


if os.path.exists(Config.MODEL_PATH):
    recognizer.read(Config.MODEL_PATH)


def predict_face(gray_face):
    if not os.path.exists(Config.MODEL_PATH):
        print("[Predict] Failed: Model file model/lbph_model.yml not found")
        return None, None, "Model not found"

    face = preprocess_face(gray_face)
    label, confidence = recognizer.predict(face)
    
    is_recognized = confidence < Config.CONFIDENCE_THRESHOLD
    status = "recognized" if is_recognized else "unknown"

    print(f"[Predict] Predicted ID: {label} | Distance/Confidence: {round(confidence, 2)} (Threshold: {Config.CONFIDENCE_THRESHOLD}) | Status: {status}")

    if is_recognized:
        return int(label), round(confidence, 2), "recognized"

    return None, round(confidence, 2), "unknown"

if __name__ == "__main__":
    success, message = train_model()
    print(message)