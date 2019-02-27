from recon.utils import grid_apply, down_sample, sobel_norm
from recon.plots.utils import compare

import numpy as np


TILE_SIZE = 16
STRIDE = 8
TEMPLATE_PATH = "/home/pushkar/scratchpad/face-rec/face-rec/recon/data/live/" \
                "images/868996030168140/collected/{timestamp}.jpeg"


if __name__ == "__main__":
    # image = np.random.normal(0, 1, (120, 160))
    image1 = down_sample(TEMPLATE_PATH.format(timestamp=1549981999000), ratio=4)
    image2 = down_sample(TEMPLATE_PATH.format(timestamp=1549994952000), ratio=4)
    image3 = down_sample(TEMPLATE_PATH.format(timestamp=1550003380000), ratio=4)
    image4 = down_sample(TEMPLATE_PATH.format(timestamp=1550004298000), ratio=4)

    applied1 = grid_apply(image1, TILE_SIZE, TILE_SIZE, STRIDE, STRIDE,
                          lambda _image: np.median(sobel_norm(_image)))
    applied2 = grid_apply(image2, TILE_SIZE, TILE_SIZE, STRIDE, STRIDE,
                          lambda _image: np.median(sobel_norm(_image)))
    applied3 = grid_apply(image3, TILE_SIZE, TILE_SIZE, STRIDE, STRIDE,
                          lambda _image: np.median(sobel_norm(_image)))
    applied4 = grid_apply(image4, TILE_SIZE, TILE_SIZE, STRIDE, STRIDE,
                          lambda _image: np.median(sobel_norm(_image)))
    compare(applied3, applied4)
