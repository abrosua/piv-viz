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
    ext = "png"  # 'png' for standard image and 'eps' for latex format; use None to disable!
    show_figure = False

    camera_fps = 22000  # INPUT the camera FPS in here!
    cal_factor = 4.0  # [mm] INPUT the pixel-to-real displacement calibration here!

    netname = "Hui-LiteFlowNet"
    imgdir = "Test 06 EDTA EA Full 22000 fps"
    labelpaths = sorted(glob(os.path.join("./labels", imgdir, "*.json")))

    #  <-------------- Calculate max velo, calibrartion factor and  colormap (Uncomment if used) -------------->
    max_velo_mag = 9.7  # [pix]
    max_velo_mag_real = 4500  # [mm/s] (use None if NOT needed!) - 3000 or 7500 for NAOCL flow

    # Vorticity/Shear stress
    max_vort_real = 32  # [1/ms] (use None if NOT needed!)
    max_shear_real = None  # [N/m2] (use None if NOT needed!)
    viscosity = 1  # Still NO reference

    calib_path = "./labels/Test 06 EDTA EA Full 22000 fps/Test 06 EDTA EA Full 22000 fps_00000.json"
    calib_label = utils.Label(calib_path, netname, verbose=1).label["calib"]
    calib_point = np.array(calib_label["points"][0])
    calib_line = np.linalg.norm(calib_point[0, :] - calib_point[1, :])

    cal_factor = cal_factor / calib_line  # [mm/pixel]
    velo_factor = camera_fps * cal_factor
    max_velo_mag_real = max_velo_mag * velo_factor if max_velo_mag_real is None else max_velo_mag_real
    factor = cal_factor / viscosity

    # utils.color_map(maxmotion=max_velo_mag, show=True)

    # <------------------------ Change of air column level (Uncomment if used) ------------------------>
    # column_level, _ = utils.column_level(labelpaths, netname, fps=13000, show=True, verbose=1)

    # <------------------------ FlowViz main script (uncomment for usage) ------------------------>
    step = 5
    cropper = (100, 0, 0, 0)
    viz_plot = utils.FlowViz(labelpaths, netname, vector_step=step, use_quiver=True, use_color=2, color_type="mag",
                             key="flow", maxmotion=max_velo_mag_real, crop_window=cropper,
                             calib=cal_factor, fps=camera_fps, verbose=1)
    # viz_plot.plot(ext=ext, show=show_figure)

    flodir = "./results/Hui-LiteFlowNet/Test 06 EDTA EA Full 22000 fps/flow"
    vid_labelpath = ["./labels/Test 06 EDTA EA Full 22000 fps/Test 06 EDTA EA Full 22000 fps_00000.json"]
    viz_video = utils.FlowViz(vid_labelpath, netname, vector_step=step, use_quiver=True, use_color=2, color_type="mag",
                              key="video", maxmotion=max_velo_mag_real, crop_window=cropper, factor=None,
                              calib=cal_factor, fps=camera_fps, verbose=1)

    # ***** Multi PLOT *****
    # viz_video.multiplot(flodir, ext=ext, start_at=0, num_images=5000)

    # ***** VIDEO *****
    viz_video.video(flodir, start_at=1800, num_images=400, fps=3, ext="mp4")
    #viz_video.video(flodir, start_at=9900, num_images=-1, fps=30, ext="mp4")
    start_id = [0.01, 0.07, 0.09, 0.3, 0.5, 1.0, 1.5]  # [s]
    for id in start_id:
        id_frame = int(camera_fps * id)
        viz_video.video(flodir, start_at=id_frame, num_images=400, fps=3, ext="mp4")

    tmpdir = "./results/Hui-LiteFlowNet/meme/flow"
    tmp_labelpath = ["./labels/meme/meme_00000.json"]
    # tmpdir = "./results/PIV-LiteFlowNet-en/test/flow"
    # tmp_labelpath = ["./labels/test/cylinder_Re40_00001_img1.json"]
    tmp_netname = os.path.basename(os.path.dirname(os.path.dirname(tmpdir)))

    max_flow, _ = utils.get_max_flow(tmpdir, tmp_labelpath[0], verbose=1)
    tmp_video = utils.FlowViz(tmp_labelpath, tmp_netname, vector_step=16, use_quiver=False, use_color=-2, color_type=None,
                              key="video", maxmotion=1.3*max_flow, verbose=1)
    tmp_video.video(tmpdir, fps=3, ext="gif", imgext="tif")


    # <------------------------ Get regional velocity (uncomment for usage) ------------------------>
    labelpath = os.path.join("./labels/Test 06 EDTA EA Full 22000 fps/.json")
    flodir = "./results/Hui-LiteFlowNet/Test 06 EDTA EA Full 22000 fps/flow"
    v2_record = utils.region_velo(labelpath, netname, flodir, key="v2", fps=13000, calibration_factor=cal_factor,
                                  start_at=0, end_at=9000, num_flows=100, avg_step=10,
                                  verbose=1)
    plt.plot(v2_record[:, 0], v2_record[:, -1])
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.show()

    print("DONE!")
