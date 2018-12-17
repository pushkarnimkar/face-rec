from flask import Flask, request, render_template, redirect

import base64
import cv2
import io
import json
import numpy as np
import os
import signal
import sys
import time

from face_rec import FaceRecognizer
from image_store import ImageStore


def exit_routine(_, __):
    recognizer.store.write()
    sys.exit(0)


with open(os.environ["FACE_REC_CONFIG"]) as config_file:
    config: dict = json.load(config_file)


app_dir = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__, static_url_path="/static",
            static_folder=os.path.join(app_dir, "static"))


info_file_path = os.path.join(config["STORE_DIR"], ImageStore.INFO_FILE_NAME)
if os.path.exists(info_file_path):
    recognizer = FaceRecognizer(store_dir=config["STORE_DIR"])
    recognizer.train()
elif "TRAIN_DIR" in config:
    recognizer = FaceRecognizer.build(config["TRAIN_DIR"], config["STORE_DIR"])
else:
    raise ValueError("insufficient input")


if not os.path.exists(config["UPLOAD_DIR"]):
    os.mkdir(config["UPLOAD_DIR"])


signal.signal(signal.SIGINT, exit_routine)


def encode_image(img, box=None, cvt_color=False):
    if cvt_color:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if box is not None:
        pt0, pt1 = (box[1], box[0]), (box[3], box[2])
        cv2.rectangle(img, pt0, pt1, (255, 0, 0), 2)

    # expects image in BGR format understood by cv2
    image_bytes = cv2.imencode(".jpg", img)[1].tostring()
    return base64.encodebytes(image_bytes).decode("ascii")


@app.route("/feed", methods=["GET", "POST"])
def feed():
    if request.method == "POST":
        image_file = request.files["image"]
        temp_file = io.BytesIO()
        image_file.save(temp_file)

        formatted = np.frombuffer(temp_file.getvalue(), dtype=np.int8)
        # cv2.imdecode gives image in BGR format understood by cv2
        image = cv2.imdecode(formatted, 1)

        # we convert image to RGB before feeding to recognizer
        name, pred = recognizer.feed(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "intangles", int(time.time())
        )

        if name is None and pred is None:
            return json.dumps(dict(status="could not detect face"))

        if isinstance(pred, dict):
            image_base64 = encode_image(image, pred["box"])
            pred["conf"] = round(float(pred["conf"]), 6)
            pred["status"] = f"predicted subject {pred['sub']} " \
                             f"with confidence {pred['conf']}"
            pred["image"] = image_base64
            return json.dumps(pred)
        else:
            return json.dumps(dict(status="unexpected error"))

    return render_template("feed.html")


@app.route("/train", methods=["GET"])
def train():
    return render_template("train.html")


@app.route("/ask", methods=["GET"])
def ask():
    tagged = recognizer.ask()

    if tagged is None:
        return json.dumps(dict(status="complete"))
    else:
        name, (image, info) = tagged

    image_base64 = encode_image(image)
    message = dict(name=name, image=image_base64, status="progress",
                   pred=info["subject"], conf=float(info["confidence"]))

    return json.dumps(message)


@app.route("/tell/<name>/<subject>", methods=["GET"])
def tell(name=None, subject=None):
    recognizer.tell(name, subject)
    return json.dumps(dict(redirect="/train"))


@app.route("/subs_list", methods=["GET"])
def list_subs():
    return json.dumps(recognizer.subs_lst)


@app.route("/retrain", methods=["GET"])
def retrain():
    recognizer.train()
    return redirect("/feed")


@app.route("/")
def hello_world():
    return redirect("/feed")
