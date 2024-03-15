from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import requests
from PIL import Image
from dotenv import dotenv_values
from flask import Flask, request
from pydantic import BaseModel
from roboflow import Roboflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

import API.database as db

DATA_DIR = Path("data")
STAMPS_DIR = DATA_DIR / "stamps"
TEMP_DATA_DIR = DATA_DIR / "temp"

MODEL_WEIGHTS_PATH = Path("Embeddings") / "model_weights.h5"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

if not MODEL_WEIGHTS_PATH.exists():
    r = requests.get(
        "https://huggingface.co/CookieByte/stamp-embeddings/resolve/main/CNN_with_new_data_final6.h5"
    )
    if r.status_code == 200:
        with MODEL_WEIGHTS_PATH.open("wb") as f:
            f.write(r.content)
    else:
        raise ValueError("Failed to get model weights from huggingface.co")

config = dotenv_values(".env")


class StampRecognition(BaseModel):
    stamp: int
    error: Optional[str]
    data: Dict[str, Any]


class ImageUploadResult(BaseModel):
    status: str
    error: Optional[str]
    data: List[StampRecognition]


class ImagesUploadResponse(BaseModel):
    data: Dict[str, ImageUploadResult]


app = Flask(__name__)

if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)

if not STAMPS_DIR.exists():
    STAMPS_DIR.mkdir(parents=True)

if not TEMP_DATA_DIR.exists():
    TEMP_DATA_DIR.mkdir(parents=True)

app.config["DATA_DIR"] = DATA_DIR
app.config["STAMPS_DIR"] = STAMPS_DIR
app.config["TEMP_DATA_DIR"] = TEMP_DATA_DIR

model = load_model(str(MODEL_WEIGHTS_PATH))

rf = Roboflow(api_key=config['API'])
project = rf.workspace("stamp-project").project("stamp-recognition-sdoan")
detection_model = project.version(3).model


def apply_threshold(image):
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return binary


def invert_image(image):
    inverted = cv2.bitwise_not = (image)
    return inverted


def cut_image(image_path, x, y, width, height):
    # Open the image
    image = Image.open(image_path)

    # Calculate the coordinates for cropping
    left = x - width // 2
    top = y - height // 2
    b = y + height // 2
    right = x + width // 2

    # Crop the image
    cropped_image = image.crop((left, top, right
                                , b))

    # Save the cropped image
    cropped_image.save(image_path)


def preprocess_image(image_path, frame):
    cut_image(image_path, frame['x'], frame['y'], frame['width'], frame['height'])
    img = cv2.imread(image_path, 0)
    img = apply_threshold(img)
    img = invert_image(img)
    img = img_to_array(img)
    img = img / 255.0
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=2)
    return img


def extract_image_embedding(img_path, frame, model):
    img = preprocess_image(img_path, frame)
    q = np.array([img])
    embedding = model.predict(q)
    return embedding.flatten()


def get_embedding(img_path, frame):
    global model
    return extract_image_embedding(img_path, frame, model)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/images/upload", methods=["POST"])
def upload_file():
    response = ImagesUploadResponse(data={})
    for name, file in request.files.items():
        if not file or file.filename == "":
            response.data[name] = ImageUploadResult(
                status="failed",
                error="no file submitted",
                data=[],
            )
            continue

        if not allowed_file(file.filename):
            response.data[name] = ImageUploadResult(
                status="failed",
                error="forbidden file",
                data=[],
            )
            continue

        # if file.filename.lower().endswith('.pdf'):

        filename = secure_filename(f"{name}_{file.filename}")
        filepath = app.config["TEMP_DATA_DIR"].joinpath(filename)
        file.save(filepath)

        detected_stamps = detection_model.predict(str(filepath), confidence=30, overlap=30).json()['predictions']

        if not detected_stamps:
            response.data[name] = ImageUploadResult(
                status="failed",
                error="no stamps found",
                data=[],
            )

        response.data[name] = ImageUploadResult(
            status="processed",
            error=None,
            data=[],
        )

        for index, frame in enumerate(detected_stamps):
            embedding = get_embedding(str(filepath), frame)
            best_match = db.find_max_cosine_similarity(embedding)

            if not best_match:
                response.data[name].data.append(
                    StampRecognition(stamp=index,
                                     error="no matching stamp found", data={})
                )
                continue

            response.data[name].data.append(
                StampRecognition(stamp=index, error=None, data=best_match)
            )

        if len(response.data[name].data) == 0:
            response.data[name].error = "no stamps identified"

    return response.json()


class EmbiddingProcessing(BaseModel):
    stamp: int
    error: Optional[str]
    status: str


class StampUploadResult(BaseModel):
    status: str
    error: Optional[str]
    data: List[EmbiddingProcessing]


class AddStampsResponse(BaseModel):
    data: Dict[str, StampUploadResult]


@app.route("/images/add_stamp", methods=["POST"])
def add_stamp():
    response = AddStampsResponse(data={})
    file_stamp_group_names_mapping = request.form

    for name, file in request.files.items():
        stamp_group_name = file_stamp_group_names_mapping[name + "_stamp_group_name"]

        related_stamps = db.find_stamp_group(stamp_group_name)

        if len(related_stamps) != 0:
            response.data[stamp_group_name] = StampUploadResult(
                status="failed",
                error="not unique stamp group name",
                data=[]
            )
            continue

        if not file or file.filename == "":
            response.data[stamp_group_name] = StampUploadResult(
                status="failed",
                error="no file submitted",
                data=[]
            )
            continue

        if not allowed_file(file.filename):
            response.data[stamp_group_name] = StampUploadResult(
                status="failed",
                error="not allowed file",
                data=[],
            )
            continue

        filename = secure_filename(f"{stamp_group_name}_{file.filename}")
        filepath = app.config["STAMPS_DIR"].joinpath(filename)
        file.save(filepath)

        detected_stamps = detection_model.predict(str(filepath), confidence=30, overlap=30).json()['predictions']

        if not detected_stamps:
            response.data[stamp_group_name] = StampUploadResult(
                status="failed",
                error="no stamps found",
                data=[],
            )

        response.data[stamp_group_name] = StampUploadResult(
            status="proccessed",
            error=None,
            data=[],
        )

        for index, frame in enumerate(detected_stamps):
            embedding = get_embedding(str(filepath), frame)

            if len(embedding) == 0:
                response.data[stamp_group_name].data.append(EmbiddingProcessing(
                    stamp=index,
                    error="no embeddings extracted",
                    status="failed"
                ))
                continue

            successful = db.insert_new_object(f"{stamp_group_name}-{index}", embedding)

            if successful:
                response.data[stamp_group_name].data.append(EmbiddingProcessing(
                    stamp=index, error=None, status="successed"
                ))
            else:
                response.data[stamp_group_name].data.append(EmbiddingProcessing(
                    stamp=index, error="not unique name", status="failed"
                ))

        if len(response.data[stamp_group_name].data) == 0:
            response.data[stamp_group_name].error = "failed to extract any embeddings"

    return response.json()


@app.route("/", methods=["GET"])
def index():
    return "OK"
