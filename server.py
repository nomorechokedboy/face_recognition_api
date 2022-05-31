import os

import cloudinary
import cloudinary.uploader
import cv2
import werkzeug.utils
from dotenv import load_dotenv
from flask import Flask
from flask import jsonify as json
from flask import request as req
from flask_cors import CORS, cross_origin

from util_funcs import (blob_to_image, get_face_image, predict_image,
                        train_model)

load_dotenv()

app = Flask(__name__)

cloudinary.config(cloud_name = os.getenv('cloud_name'), api_key = os.getenv('api_key'), api_secret = os.getenv('api_secret'))

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def index():
    return json(
        message = "lmao burh burh"
    )

@app.route("/student/upload", methods=["POST", "PUT"])
def upload():
    data = req.files["video"]
    dirname = req.form["name"]

    filename = werkzeug.utils.secure_filename(str(data.filename))

    saved_directory = f"./datasets/{dirname}"

    if not os.path.exists(saved_directory):
        os.mkdir(saved_directory)

    data.save(f"./videos/{filename}")
    
    get_face_image(dirname)
    train_model()

    return json(
        message = "Video upload successfully"
    )

@app.route('/predict', methods=['POST'])
def handler():
    data = req.files["image"]
    filename = './predict/predicted.jpg'

    result = predict_image(blob_to_image(data.read()))
    cv2.imwrite(filename, result)
    upload_image = cloudinary.uploader.upload(filename)
    
    return json(
        upload_image
    )
