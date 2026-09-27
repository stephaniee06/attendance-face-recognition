# Face Attendance System

A web-based attendance system that recognises faces from a browser webcam. The backend is a Flask API that detects faces with OpenCV Haar cascades, identifies them with an OpenCV **LBPH** (Local Binary Patterns Histograms) face recognizer, and records attendance in a SQLite database. The frontend is a single page for signing in, registering people, capturing face photos, training the model and taking attendance.

## Features

- **Account login / registration** for the operator of the web page (email + password, hashed with Werkzeug).
- **User management:** create a person (name + optional department), list them with their photo count, delete them along with their photos and attendance records.
- **Face capture:** each click on *Capture Face* takes one webcam frame, crops the detected face, and saves it to that person's dataset folder.
- **Model training** from the saved photos, with an 80/20 train/test split per person and simple data augmentation.
- **Incremental updates:** once a model exists, each newly captured face is also added to it immediately (`LBPHFaceRecognizer.update`), so a full retrain isn't needed after every capture.
- **Recognition & attendance:** *Recognize Now* identifies the face in the current frame and, if it's a known person, writes an attendance record.
- **Offline evaluation script** that reports accuracy, precision, recall and F1 on the held-out 20%.

## How it works

```mermaid
flowchart LR
    A[Browser webcam] -->|JPEG frame, base64| B[Flask API]
    B --> C[Haar cascade<br/>face detection]
    C --> D[Crop, resize 100×100,<br/>CLAHE]
    D -->|register-face| E[(dataset/&lt;id&gt;_&lt;Name&gt;/n.jpg)]
    E -->|train| F[LBPH model<br/>model/lbph_model.yml]
    D -->|recognize| F
    F -->|distance &lt; 120| G[(SQLite: attendance)]
```

**Detection** (`face_engine/detector.py`) — The frame is converted to greyscale and passed to OpenCV's `haarcascade_frontalface_default.xml` (`scaleFactor=1.1`, `minNeighbors=3`, `minSize=30×30`). The API uses the first face the detector returns.

**Pre-processing** (`face_engine/recognizer.py`) — Faces are resized to 100×100 and contrast-normalised with CLAHE (`clipLimit=2.0`, `tileGridSize=8×8`).

**Training** — For every folder in `dataset/` whose ID belongs to a user in the database:

1. Image files are sorted by name; the last 20% (at least one) are held out for testing, the rest are used for training. Folders with fewer than 2 images are skipped.
2. Each training image produces 4 samples: the original, a horizontal flip, a brighter copy and a darker copy.
3. An LBPH recognizer (`radius=1`, `neighbors=8`, `grid_x=8`, `grid_y=8`) is trained from scratch and saved to `model/lbph_model.yml`.

Example: with 5 photos of one person, `int(5 × 0.2) = 1` photo is held out, 4 are used, and the model is trained on 4 × 4 = 16 samples.

**Recognition** — LBPH returns a *distance*, not a probability: **lower means a closer match**. The code (and the UI) calls this value "confidence". A face is accepted when the distance is below `CONFIDENCE_THRESHOLD` (default `120`); otherwise it is reported as `unknown`. Every accepted recognition inserts one row into the `attendance` table.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-CORS, Flask-SQLAlchemy |
| Computer vision | OpenCV (`opencv-contrib-python-headless`, needed for `cv2.face`), NumPy |
| Database | SQLite (`instance/attendance.db`) |
| Frontend | React 18 + `htm` from unpkg, `getUserMedia` webcam API |
| Evaluation (optional) | scikit-learn |
| Production server | Gunicorn |

## Project structure

```
attendance-face-recognition/
├── app.py                  # Flask app: config, DB setup, "/" and "/camera" routes
├── config.py               # Paths, face size, recognition threshold
├── requirements.txt
├── vercel.json             # Rewrites "/" to the static camera page
├── face_engine/
│   ├── detector.py         # Haar-cascade detection, base64 → image decoding
│   └── recognizer.py       # Pre-processing, augmentation, train / update / predict, evaluation
├── models/
│   └── database.py         # SQLAlchemy models: User, Attendance
├── routes/
│   └── api.py              # REST API blueprint mounted at /api
└── templates/
    └── camera.html         # Single-page web interface
```

Created at runtime (and git-ignored):

```
instance/attendance.db      # users and attendance records
instance/auth_users.json    # operator accounts (hashed passwords, session tokens)
dataset/<id>_<Name>/1.jpg … # captured face crops, one folder per person
model/lbph_model.yml        # trained LBPH model
```

## Usage

1. **Sign up / log in** on the start screen.
2. **Create a user** — enter a name (and optionally a department) and click *Create User*. A folder `dataset/<id>_<Name>/` is created.
3. **Capture faces** — select the user from the dropdown, face the camera and click *Capture Face* several times, varying angle and lighting slightly. At least 2 photos are required; more gives the model more to learn from.
4. **Train** — click *Train Model*. Retrain after deleting users or capturing many new photos.
5. **Take attendance** — click *Recognize Now*. A recognised face is logged in the database with a UTC timestamp and status `Hadir` ("present").

