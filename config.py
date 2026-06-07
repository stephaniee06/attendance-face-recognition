import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)


class Config:
    SQLALCHEMY_DATABASE_URI = (
        "sqlite:///" + os.path.join(INSTANCE_DIR, "attendance.db")
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    DATASET_DIR = DATASET_DIR

    MODEL_DIR = MODEL_DIR

    MODEL_PATH = os.path.join(MODEL_DIR, "lbph_model.yml")

    FACE_SIZE = (100, 100)

    CONFIDENCE_THRESHOLD = 75