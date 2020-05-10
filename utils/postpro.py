import sys
import os
import imageio
import json
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
from labelme.utils import shape_to_mask
from flowviz import animate, colorflow

import matplotlib
matplotlib.use('TkAgg')

# Manage the working directory
maindir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(maindir)

# Importing from LiteFlowNet
pivdir = os.path.join(os.path.dirname(maindir), "thesis_faber")
sys.path.append(pivdir)
from src.utils_plot import read_flow, read_flow_collection


def use_flowviz(flodir, imdir, start_at: int = 0, num_images: int = -1, lossless: bool = True):
    """
    Visualization using flowviz module
    """
    print("Optical flow visualization using flowviz by marximus")

    # Obtaining the flo files and file basename
    flows, flonames = read_flow_collection(flodir, start_at=start_at, num_images=num_images)
    fname_addition = f"-{start_at}_all" if num_images < 0 else f"-{start_at}_{num_images}"
    fname = os.path.basename(flodir).rsplit('-', 1)[0] + fname_addition

    # Manage the input images
    video_list = []
    for floname in flonames:
        filename = os.path.basename(floname).rsplit('_', 1)[0] + ".tif"
        filepath = os.path.join(imdir, filename)
        video_list.append(imageio.imread(filepath))
        # video_list.append(Image.open(filepath).convert('RGB'))

    video = np.array(video_list)

    # Previewing the files
    print('Video shape: {}'.format(video.shape))
    print('Flows shape: {}'.format(flows.shape))

    # Define the output files
    colors = colorflow.motion_to_color(flows)
    flowanim = animate.FlowAnimation(video=video, video2=colors, vector=flows, vector_step=10,
                                     video2_alpha=0.5, scale=0.5)

    # Saving the video
    viddir = os.path.join(os.path.dirname(flodir), "videos")
    if not os.path.isdir(viddir):
        os.makedirs(viddir)

    if lossless:
        vidpath = os.path.join(viddir, fname + ".avi")
        flowanim.save(vidpath, codec="ffv1")
    else:
        vidpath = os.path.join(viddir, fname + ".mp4")
        flowanim.save(vidpath)
    print(f"Finish saving the video file ({vidpath})!")


def get_label(flopaths: List[str,], labeldir: str = './labels'):
    labels = {}
    print(os.getcwd())

    # Iterate over all the input flo paths
    for flo in flopaths:
        label_name = os.path.basename(flo).rsplit('_', 1)[0] + ".json"
        basedir = os.path.basename(os.path.dirname(flo)).rsplit('-', 1)[0]
        label_path = os.path.join(labeldir, basedir, label_name)

        with open(label_path) as json_file:
            data = json.load(json_file)
            img_shape = [data['imageHeight'], data['imageWidth']]
            label = {}

            for d in data['shapes']:
                mask = shape_to_mask(img_shape, d['points'], shape_type=d['shape_type'])
                label[d['label']] = {'flo': _masked_flo(flo, mask), 'mask': mask}

        labels[flo] = label
    return labels


def _masked_flo(flopath, mask, fill_with: Optional[float] = None):
    # Initialization
    out_flow = read_flow(flopath)
    mask_flow = np.empty(out_flow.shape)
    mask_flow[:] = np.nan if fill_with is None else fill_with

    # Filling the masked flow array
    mask_flow[mask] = out_flow[mask]

    return mask_flow


def velo_mean(flo: np.array, mask: Optional[np.array] = None):
    if mask is None:
        flo_mag = np.linalg.norm(flo, axis=-1)
        flo_mag_clean = flo_mag[~np.isnan(flo_mag)]

    else:
        flo_clean = flo[mask]
        flo_mag_clean = np.linalg.norm(flo_clean, axis=-1)

    return np.mean(flo_mag_clean)


if __name__ == '__main__':
    # <------------ Use flowviz (uncomment for usage) ------------>
    # flodir = "./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif-0_end"
    # imdir = "./frames/Test 03 L3 NAOCL 22000 fpstif"
    # use_flowviz(flodir, imdir, start_at=4900, num_images=50, lossless=False)

    # <------------ Use get_label (uncomment for usage) ------------>
    flodir = "./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif-0_end"
    floidx = [4900]

    flopaths = []
    for idx in floidx:
        fname = os.path.basename(flodir).rsplit('-', 1)[0] + f"_{str(idx).zfill(5)}_out.flo"
        flopaths.append(os.path.join(flodir, fname))

    # Getting the label
    labels = get_label(flopaths)
    v1 = labels[flopaths[0]]['v1']
    v2 = labels[flopaths[0]]['v2']
    flow = labels[flopaths[0]]['flow']

    # Data post-processing (i.e., calculate mean, deviation, etc)
    # v1_mean = velo_mean(v1['flo'], mask=v1['mask'])

    print('DONE!')
