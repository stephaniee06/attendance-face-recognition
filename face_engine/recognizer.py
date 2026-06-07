import os
import cv2
import numpy as np

from config import Config
from face_engine.detector import detect_faces

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=16,
    grid_x=10,
    grid_y=10
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
    aligned = align_face(gray_face)
    resized = cv2.resize(aligned, Config.FACE_SIZE)
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
        return False, "Dataset masih kosong"

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


# ==============================================================================
# KODE TAMBAHAN UNTUK EVALUASI METRIK (AUTO-SPLIT TANPA FOLDER BARU)
# ==============================================================================

def evaluate_model_metrics(test_dataset_dir=None):
    """
    Fungsi untuk mengevaluasi performa model secara otomatis.
    Mengambil 20% data dari folder dataset utama sebagai data uji (testing data).
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    # Langsung menembak folder dataset utama kamu
    target_dir = Config.DATASET_DIR
        
    print(f"\n==================================================")
    print(f"MEMULAI EVALUASI MODEL (Auto-Split 20% Data Uji)")
    print(f"Folder Sumber: {target_dir}")
    print(f"==================================================")
    
    if not os.path.exists(target_dir):
        print(f"[Error] Folder dataset '{target_dir}' tidak ditemukan!")
        return None
        
    if not os.path.exists(Config.MODEL_PATH):
        print(f"[Error] Model lbph_model.yml tidak ditemukan. Jalankan train dulu.")
        return None

    y_true = []
    y_pred = []
    
    folders = os.listdir(target_dir)
    
    # Membaca setiap folder kelas user secara eksplisit dengan loop biasa
    for folder_name in folders:
        user_folder = os.path.join(target_dir, folder_name)
        if not os.path.isdir(user_folder):
            continue
            
        try:
            actual_id = int(folder_name.split("_")[0])
        except ValueError:
            continue
            
        # Mengumpulkan semua file gambar yang valid dalam folder
        all_images = []
        for f in os.listdir(user_folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(f)
        
        if len(all_images) < 2:
            print(f"[Skip Uji] Folder '{folder_name}' memiliki foto terlalu sedikit untuk dievaluasi.")
            continue
            
        # Pisahkan 20% foto terakhir dari list untuk dijadikan bahan testing
        test_size = max(1, int(len(all_images) * 0.2))
        test_images = all_images[-test_size:]
        
        print(f"Menguji {len(test_images)} file foto terakhir untuk User ID: {actual_id} ({folder_name})")
        
        # Memproses file testing secara eksplisit dengan loop biasa
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
            else:
                print(f"  [Gagal Deteksi] Wajah pada file '{filename}' tidak terdeteksi saat uji.")

    if len(y_true) == 0:
        print("\n[Error] Tidak ada wajah dari data uji yang berhasil diproses.")
        return None

    # Kalkulasi nilai metrik evaluasi
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n==================================================")
    print(f"HASIL EVALUASI METRIK PENGENALAN WAJAH")
    print(f"==================================================")
    print(f"Accuracy  (): {round(acc, 4)}  (atau {round(acc * 100, 2)}%)")
    print(f"Precision (): {round(prec, 4)}  (atau {round(prec * 100, 2)}%)")
    print(f"Recall    (): {round(rec, 4)}  (atau {round(rec * 100, 2)}%)")
    print(f"F1-Score  (): {round(f1, 4)}  (atau {round(f1 * 100, 2)}%)")
    print(f"==================================================")
    
    print("\nLaporan Detail per Kelas/User ID:")
    print("(Catatan: ID -1 adalah representasi wajah yang tertebak sebagai 'unknown')")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }


if __name__ == "__main__":
    success, message = train_model()
    print(message)
    
    # Menjalankan fungsi evaluasi otomatis menggunakan metode Auto-Split
    evaluate_model_metrics()