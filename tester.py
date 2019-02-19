from app.face_rec import FaceRecognizer

import sys


def main(store_dir: str):
    recognizer = FaceRecognizer(store_dir=store_dir)
    recognizer.train()
    recognizer.feed_again()
    recognizer.store.write()


if __name__ == "__main__":
    main(sys.argv[1])
