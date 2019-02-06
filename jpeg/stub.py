from argparse import ArgumentParser
from jpeg.compressed import Compressed
from jpeg.methods import decompress_mcus
from matplotlib import pyplot as plt


def main(stream):
    compressed = Compressed(stream)
    compressed.parse()
    quant_tbl = compressed.scan_headers[0].components[1].quant_tbl
    buffer = decompress_mcus(compressed.mcus[1], 0, 0, 60, 80, quant_tbl)
    plt.imshow(buffer)
    plt.show()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image", help="location of image to be read")
    args = parser.parse_args()
    main(open(args.image, "rb"))
