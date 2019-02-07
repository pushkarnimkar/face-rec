from argparse import ArgumentParser
from jpeg.compressed import Compressed
from matplotlib import pyplot as plt


def main(stream):
    compressed = Compressed(stream)
    compressed.parse()
    buffer = compressed.decompress_region(0, 0, 60, 80)
    plt.imshow(buffer, cmap="gray")
    plt.show()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image", help="location of image to be read")
    args = parser.parse_args()
    main(open(args.image, "rb"))
