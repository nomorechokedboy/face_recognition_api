import base64
import io
from pyexpat import model

import cv2
from flask import Flask, request as req, jsonify as json, send_file, make_response as make_res
import os
import werkzeug.utils
from util_funcs import blob_to_image, predict_image, predict_image, get_face_image, train_model
from flask_cors import CORS, cross_origin
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
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
