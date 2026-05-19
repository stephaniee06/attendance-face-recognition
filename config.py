import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SQLALCHEMY_DATABASE_URI = \
        "sqlite:///" + os.path.join(
            BASE_DIR,
            "instance",
            "attendance.db"
        )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    DATASET_DIR = os.path.join(
        BASE_DIR,
        "dataset"
    )

    MODEL_DIR = os.path.join(
        BASE_DIR,
        "model"
    )

    MODEL_PATH = os.path.join(
        MODEL_DIR,
        "lbph_model.yml"
    )

    FACE_SIZE = (200,200)

    CONFIDENCE_THRESHOLD = 70