from flask import Flask, request, render_template, redirect
from recon.app.device import USBCamera
from recon.app.face_rec import FaceRecognizer

import base64
import cv2
import io
import json
import os
import serial
import signal
import sys
import time


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


def exit_routine(_, __):
    recognizer.store.write()
    global camera
    camera.close()
    sys.exit(0)


with open(CONFIG_FILE) as config_file:
    config: dict = json.load(config_file)

    app_dir = os.path.dirname(os.path.realpath(__file__))
    app = Flask(__name__, static_url_path="/static",
                static_folder=os.path.join(app_dir, "static"))

    recognizer = FaceRecognizer(store_dir=config["STORE_DIR"])
    recognizer.train()
    camera = USBCamera()
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
        temp_file = io.BytesIO()
        request.files["image"].save(temp_file)
        buffer = temp_file.getvalue()
        name, pred = recognizer.feed(buffer, "intangles", int(time.time()))

        if name is None and pred is None:
            return json.dumps(dict(status="could not detect face"))

        if isinstance(pred, dict):
            image_base64 = encode_image(pred["image"], pred["box"],
                                        cvt_color=True)
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


@app.route("/acquire")
def acquire():
    global camera
    try:
        image = camera.acquire_image()
        if image is None:
            raise serial.SerialException()
        image_base64 = encode_image(image)
        status = "acquired image from device"
        return json.dumps(dict(image=image_base64, status=status))
    except serial.SerialException as se:
        print(se, sys.stderr)
        status = "failed to acquire image"
        return json.dumps(dict(status=status))


@app.route("/")
def hello_world():
    return redirect("/feed")


@app.route("/feed-again")
def feed_again():
    recognizer.feed_again()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
