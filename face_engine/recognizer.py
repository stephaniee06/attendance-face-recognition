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


# --- LOGIKA BARU: SEPARASI TRAIN & TEST DATA SECARA KETAT ---
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
            continue
        
        if valid_user_ids is not None and user_id not in valid_user_ids:
            continue

        print(f"\nProcessing Folder untuk Training: '{folder_name}' (User ID: {user_id})")
        
        # Ambil semua file gambar
        all_images = [f for f in os.listdir(user_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        all_images.sort() # urutkan agar konsisten dengan split evaluasi
        
        if len(all_images) < 2:
            print(f"[Skip] File terlalu sedikit untuk dipisah Train-Test split.")
            continue
            
        # LOGIKA PERBAIKAN: Sisakan 20% data paling belakang untuk Testing.
        # Ambil hanya 80% data depan untuk dimasukkan ke proses training model.
        test_size = max(1, int(len(all_images) * 0.2))
        train_images = all_images[:-test_size] 

        user_faces_count = 0
        user_augmented_count = 0

        for filename in train_images:
            image_path = os.path.join(user_folder, filename)
            image = prepare_training_face(image_path)

            if image is None:
                skipped += 1
                continue

            faces.append(image)
            labels.append(user_id)
            user_faces_count += 1

            # Augmentasi hanya boleh dilakukan pada data training (80% tadi)
            for aug_face in augment_face(image):
                faces.append(aug_face)
                labels.append(user_id)
                user_augmented_count += 1

        print(f"Result Train Split '{folder_name}': {user_faces_count} foto original dimasukkan ke training model.")

    if len(faces) == 0:
        return False, "Dataset is empty or no valid faces detected"

    print("\nTraining model on 80% split data...")
    recognizer.train(faces, np.array(labels))

    print(f"Model saved to: {Config.MODEL_PATH}")
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    recognizer.save(Config.MODEL_PATH)
    print("Training Completed & Model Successfully Saved\n")

    return True, f"Model trained with {len(faces)} photos (augmented included)"


def update_model(gray_face, user_id):
    if not os.path.exists(Config.MODEL_PATH):
        return False, "Model not trained. Please train the model first."

    face = preprocess_face(gray_face)
    faces = [face] + augment_face(face)
    labels = np.array([user_id] * len(faces))

    recognizer.update(faces, labels)
    recognizer.save(Config.MODEL_PATH)
    return True, f"Model updated with {len(faces)} faces for user ID {user_id}"


if os.path.exists(Config.MODEL_PATH):
    recognizer.read(Config.MODEL_PATH)


def predict_face(gray_face):
    if not os.path.exists(Config.MODEL_PATH):
        return None, None, "Model not found"

    face = preprocess_face(gray_face)
    label, confidence = recognizer.predict(face)
    
    is_recognized = confidence < Config.CONFIDENCE_THRESHOLD
    status = "recognized" if is_recognized else "unknown"

    print(f"[Predict] Predicted ID: {label} | Distance/Confidence: {round(confidence, 2)} | Status: {status}")

    if is_recognized:
        return int(label), round(confidence, 2), "recognized"

    return None, round(confidence, 2), "unknown"


def evaluate_model_metrics(test_dataset_dir=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    target_dir = Config.DATASET_DIR
        
    print(f"\n==================================================")
    print(f"MEMULAI EVALUASI MODEL (Menguji 20% Data Rahasia)")
    print(f"==================================================")
    
    if not os.path.exists(target_dir) or not os.path.exists(Config.MODEL_PATH):
        print("[Error] Dataset atau berkas model tidak ditemukan.")
        return None

    y_true = []
    y_pred = []
    
    folders = os.listdir(target_dir)
    
    for folder_name in folders:
        user_folder = os.path.join(target_dir, folder_name)
        if not os.path.isdir(user_folder):
            continue
            
        try:
            actual_id = int(folder_name.split("_")[0])
        except ValueError:
            continue
            
        all_images = [f for f in os.listdir(user_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        all_images.sort()
        
        if len(all_images) < 2:
            continue
            
        # Ambil 20% data paling belakang (data yang tidak disentuh oleh training tadi)
        test_size = max(1, int(len(all_images) * 0.2))
        test_images = all_images[-test_size:]
        
        print(f"Menguji {len(test_images)} file foto rahasia untuk User ID: {actual_id} ({folder_name})")
        
        for filename in test_images:
            image_path = os.path.join(user_folder, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
                
            faces, gray = detect_faces(image)
            
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                face_crop = gray[y:y+h, x:x+w]
                
                predicted_id, confidence, status = predict_face(face_crop)
                
                if status == "recognized" and predicted_id is not None:
                    final_prediction = predicted_id
                else:
                    final_prediction = -1 
                
                y_true.append(actual_id)
                y_pred.append(final_prediction)

    if len(y_true) == 0:
        print("\n[Error] Tidak ada data uji yang berhasil diproses.")
        return None

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n==================================================")
    print(f"HASIL EVALUASI METRIK PENGENALAN WAJAH (Jujur & Valid)")
    print(f"==================================================")
    print(f"Accuracy  (): {round(acc, 4)}  (atau {round(acc * 100, 2)}%)")
    print(f"Precision (): {round(prec, 4)}  (atau {round(prec * 100, 2)}%)")
    print(f"Recall    (): {round(rec, 4)}  (atau {round(rec * 100, 2)}%)")
    print(f"F1-Score  (): {round(f1, 4)}  (atau {round(f1 * 100, 2)}%)")
    print(f"==================================================")
    
    print("\nLaporan Detail per Kelas/User ID:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


if __name__ == "__main__":
    success, message = train_model()
    print(message)
    evaluate_model_metrics()