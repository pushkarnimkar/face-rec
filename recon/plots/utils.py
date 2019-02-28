from matplotlib import pyplot as plt

import numpy as np


def compare(image1, image2, bins1=None, bins2=None):
    _, ((ax1, ax2), (ax3, ax4)) = \
        plt.subplots(2, 2, sharex="row", sharey="row")
    ax1.imshow(image1, cmap="gray")
    ax2.imshow(image2, cmap="gray")
    bins = np.arange(-0.5, 256.5, 1)

    if bins1 is None:
        bins1 = bins
    if bins2 is None:
        bins2 = bins

    ax3.hist(image1.flatten(), bins=bins1)
    ax4.hist(image2.flatten(), bins=bins2)
    plt.show()
