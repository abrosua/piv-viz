import os
import sys
import imageio

import numpy as np
import matplotlib.pyplot as plt

import utils  # Import the utils directory as module

import matplotlib
matplotlib.use('TkAgg')


if __name__ == "__main__":
    # INPUT
    flopath = "./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif-0_end/Test 03 L3 NAOCL 22000 fpstif_04900_out.flo"
    impath = "./frames/Test 03 L3 NAOCL 22000 fpstif/Test 03 L3 NAOCL 22000 fpstif_04900.tif"

    labels = utils.get_label([flopath])
    flow = labels[flopath]['flow']

    # Image masking
    masked_img = imageio.imread(impath)
    masked_img[flow['mask']] = 0
    plt.imshow(masked_img)
    plt.show()

    print("DONE!")
