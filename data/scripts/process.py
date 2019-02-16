from argparse import ArgumentParser

import cv2
import face_recognition
import numpy as np
import os
import pandas as pd


def process_one_image(image: np.ndarray) -> dict:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    count_hog = len(face_recognition.face_locations(image_rgb, model="hog"))
    count_cnn = len(face_recognition.face_locations(image_rgb, model="cnn"))
    blur_score = cv2.Laplacian(image, cv2.CV_64F).var()
    return dict(count_hog=count_hog, count_cnn=count_cnn, 
                blur_score=blur_score)


def main(args):
    image_info_objects = []
    for image_file in os.listdir(args.imgdir):
        print(image_file.split("."))
        timestamp = int(image_file.split(".")[0])
        image = cv2.imread(os.path.join(args.imgdir, image_file))
        image_info = process_one_image(image)
        image_info["timestamp"] = timestamp
        image_info_objects.append(image_info)
    pd.DataFrame(image_info_objects).to_csv(args.output, index=False)


DESCRIPTION = """this generates csv file containing information about various 
                 examined image parameters of each of the sampled file"""
IMGDIR_HELP = """directory containing images to be processed"""
OUTPUT_HELP = """destination to export information of each image as csv file"""


if __name__ == "__main__":
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument("imgdir", help=IMGDIR_HELP)
    parser.add_argument("output", help=OUTPUT_HELP)
    main(parser.parse_args())

