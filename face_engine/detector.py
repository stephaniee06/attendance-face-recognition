import cv2
import numpy as np
import base64

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(80, 80)
    )

    return faces, gray

def decode_base64_frame(base64_string):
    image_data = base64.b64decode(base64_string.split(",")[1])
    np_array = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    return frame