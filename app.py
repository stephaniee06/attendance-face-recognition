from flask import Flask, render_template
from flask_cors import CORS
from routes.api import api

from config import Config
from models.database import db

app = Flask(__name__)
app.config.from_object(Config)

CORS(app)

db.init_app(app)
app.register_blueprint(api, url_prefix="/api")

with app.app_context():
    db.create_all()

@app.route("/")
def home():
    return "Backend Face Attendance System is running. Access the camera interface at /camera"

@app.route("/camera")
def camera():
    return render_template("camera.html")

if __name__ == "__main__":
    app.run(debug=True)

