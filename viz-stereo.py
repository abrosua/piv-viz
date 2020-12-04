import os
from glob import glob
import json

import numpy as np
import matplotlib.pyplot as plt
import utils  # Import the utils directory as module

import matplotlib
matplotlib.use('TkAgg')


if __name__ == "__main__":
    # <------------------------------------ INPUT ------------------------------------>
    ext = None  # 'png' for standard image and 'eps' for latex format; use None to disable!
    show_figure = True

    camera_fps = 1

    netname = "PIV-LiteFlowNet-en"
    imgdir = "test-stereo"
    flodir = "../work/results/PIV-LiteFlowNet-en/test-stereo/flow/left"
    workdir = "../work"
    labelpaths = sorted(glob(os.path.join(workdir, "labels", imgdir, "*.json")))

    #  <-------------- Calculate max velo, calibrartion factor and  colormap (Uncomment if used) -------------->
    max_velo_mag_real = None  # [mm/s] (use None if NOT needed!) - 3000 or 7500 for NAOCL flow

    # Vorticity/Shear stress
    max_vort_real = 32  # [1/ms] (use None if NOT needed!)
    max_shear_real = None  # [N/m2] (use None if NOT needed!)
    viscosity = 1  # Still NO reference

    # *** NOTE: Real measurement calibration is already done during the stereoscopic reconstruction process!
    calib_file = os.path.join(utils.split_abspath(flodir, cutting_path="flow")[0], "30-5.json")
    with open(calib_file) as fp:
        cal_factor = json.load(fp)["calib"]
    postpro_factor = cal_factor / viscosity

    # utils.color_map(maxmotion=max_velo_mag, show=True)

    # <------------------------ Change of air column level (Uncomment if used) ------------------------>
    # column_level, _ = utils.column_level(labelpaths, netname, fps=13000, show=True, verbose=1)

    # <------------------------ FlowViz main script (uncomment for usage) ------------------------>
    step = 5
    cropper = 0 # (100, 0, 0, 0); Default value is 0
    viz_plot = utils.FlowViz(labelpaths, flodir, vector_step=step, use_quiver=True, use_color=2, color_type="mag",
                             key="flow", maxmotion=max_velo_mag_real, crop_window=cropper, verbose=1,
                             use_stereo=False)
    viz_plot.plot(ext=ext, show=show_figure)

    vid_labelpath = [os.path.join(workdir, "labels", imgdir, "30-5-000.json")]
    viz_video = utils.FlowViz(vid_labelpath, flodir, vector_step=step, use_quiver=True, use_color=2, color_type="mag",
                              key="flow", maxmotion=max_velo_mag_real, crop_window=cropper, post_factor=None,
                              verbose=1)

    # ***** Multi PLOT *****
    # viz_video.multiplot(ext=ext, start_at=0, num_images=5000)

    # ***** VIDEO *****
    viz_video.video(start_at=1800, num_images=400, fps=3, ext="mp4")
    #viz_video.video(start_at=9900, num_images=-1, fps=30, ext="mp4")
    start_id = [0.01, 0.07, 0.09, 0.3, 0.5, 1.0, 1.5]  # [s]
    for id in start_id:
        id_frame = int(camera_fps * id)
        viz_video.video(start_at=id_frame, num_images=400, fps=3, ext="mp4")

    tmpdir = "./results/Hui-LiteFlowNet/meme/flow"
    tmp_labelpath = ["./labels/meme/meme_00000.json"]
    # tmpdir = "./results/PIV-LiteFlowNet-en/test/flow"
    # tmp_labelpath = ["./labels/test/cylinder_Re40_00001_img1.json"]

    max_flow, _ = utils.get_max_flow(tmpdir, tmp_labelpath[0], verbose=1)
    tmp_video = utils.FlowViz(tmp_labelpath, tmpdir, vector_step=16, use_quiver=False, use_color=-2, color_type=None,
                              key="video", maxmotion=1.3*max_flow, verbose=1)
    tmp_video.video(fps=3, ext="gif")


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
