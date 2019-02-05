from argparse import ArgumentParser
from jpeg.compressed import Compressed
import numpy as np


def main(stream):
    compressed = Compressed(stream)
    compressed.parse()
    intensities = np.array(list(map(lambda x: x[0], compressed.mcus[1])))
    img = intensities.reshape(60, 80)
    np.save("/tmp/img.npy", img)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image", help="location of image to be read")
    args = parser.parse_args()
    main(open(args.image, "rb"))
