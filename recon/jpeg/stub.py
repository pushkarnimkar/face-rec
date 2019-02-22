from argparse import ArgumentParser
from recon.jpeg.compressed import Compressed
from matplotlib import pyplot as plt

import cv2


def main(filename: str):
    compressed = Compressed(open(filename, "rb"))
    compressed.parse()
    buffer = compressed.decompress_region(0, 0, 60, 80)

    import numpy as np
    buffer = (buffer + 128).clip(0, 255).astype(np.uint8)

    _, (ax1, ax2) = plt.subplots(1, 2, sharex="all", sharey="all")
    ax1.imshow(buffer, cmap="gray")

    original = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2YCrCb)
    ax2.imshow(original[:, :, 0], cmap="gray")

    plt.show()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image", help="location of image to be read")
    args = parser.parse_args()
    main(args.image)
