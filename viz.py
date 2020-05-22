import os
from glob import glob
import imageio
from typing import Optional, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import utils  # Import the utils directory as module

import matplotlib
matplotlib.use('TkAgg')


if __name__ == "__main__":
    # INPUT
    ext = None  # 'png' for standard image and 'eps' for latex format; use None to disable!
    show_figure = True
    netname = "Hui-LiteFlowNet"
    labelpaths = sorted(glob(os.path.join("./labels", "Test 03 L3 NAOCL 22000 fpstif", "*.json")))

    # Main
    cropper = (100, 0, 0, 0)
    flow_visualizer = utils.FlowViz(labelpaths, file_extension=ext, show=show_figure, verbose=0)
    flow_visualizer(netname, vector_step=5, use_quiver=True, use_color=True, crop_window=cropper)
    # utils.color_map(maxmotion=4, show=True)

    print("DONE!")
