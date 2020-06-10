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
    # <------------------------------------ INPUT ------------------------------------>
    ext = None  # 'png' for standard image and 'eps' for latex format; use None to disable!
    show_figure = True

    camera_fps = 1  # INPUT the camera FPS in here!
    cal_factor = 1.0  # INPUT the pixel-to-real displacement calibration here!
    velo_factor = camera_fps * cal_factor

    netname = "Hui-LiteFlowNet"
    imgdir = "Test 03 L3 NAOCL 22000 fpstif"  # Test 03 L3 EDTA 22000 fpstif
    labelpaths = sorted(glob(os.path.join("./labels", imgdir, "*.json")))

    #  <------------------------ Create colormap (Uncomment if used) ------------------------>
    max_velo_mag = 9.7
    # utils.color_map(maxmotion=max_velo_mag, show=True)

    # <------------------------ Change of air column level (Uncomment if used) ------------------------>
    # column_level, _ = utils.column_level(labelpaths, netname, fps=13000, show=True, verbose=1)

    # <------------------------ FlowViz main script (uncomment for usage) ------------------------>
    step = 5
    cropper = (100, 0, 0, 0)
    viz_plot = utils.FlowViz(labelpaths, netname, vector_step=step, use_quiver=True, use_color=2, color_type="mag",
                             maxmotion=max_velo_mag, crop_window=cropper, velocity_factor=velo_factor,
                             verbose=1)
    viz_plot.plot(file_extension=ext, show=show_figure)

    flodir = "./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif/flow"
    vid_labelpath = ["./labels/Test 03 L3 NAOCL 22000 fpstif/Test 03 L3 NAOCL 22000 fpstif_13124.json"]
    viz_video = utils.FlowViz(vid_labelpath, netname, vector_step=step, use_quiver=True, use_color=2, color_type="mag",
                              key="video", maxmotion=max_velo_mag, crop_window=cropper, velocity_factor=velo_factor,
                              verbose=1)
    viz_video.video(flodir, start_at=4900, num_images=50, fps=2, ext="mp4")


    # <------------------------ Get regional velocity (uncomment for usage) ------------------------>
    labelpath = os.path.join("./labels/Test 03 L3 NAOCL 22000 fpstif/Test 03 L3 NAOCL 22000 fpstif_04900.json")
    flodir = "./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif/flow"
    v2_record = utils.region_velo(labelpath, netname, flodir, key="v2", fps=13000, calibration_factor=cal_factor,
                                  start_at=0, end_at=9000, num_flows=100, avg_step=10,
                                  verbose=1)
    plt.plot(v2_record[:, 0], v2_record[:, -1])
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.show()

    print("DONE!")
