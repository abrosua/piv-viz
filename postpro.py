import sys
import os
from typing import Optional, List
import imageio
import json

from PIL import Image
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from labelme.utils import shapes_to_label, shape_to_mask

import matplotlib
matplotlib.use('TkAgg')

from flowviz import animate, colorflow

pivdir = "../thesis_faber"
sys.path.append(pivdir)
from src.utils_plot import read_flow, read_flow_collection, motion_to_color


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


def singleplot(floname, imdir, use_flowviz=False):
    # Generate image name from the flo file
    basename = str(os.path.basename(floname).rsplit('_', 1)[0])
    subdir = os.path.basename(os.path.dirname(floname)).rsplit('-', 1)[0]
    imname = os.path.join(imdir, subdir, basename + ".tif")

    # Generating the data
    image = imageio.imread(imname)
    out_flow = read_flow(floname)

    # plotting image
    if use_flowviz:
        flow = out_flow.reshape([1, *out_flow.shape])
        colors = colorflow.motion_to_color(flow)
        flowanim = animate.FlowAnimation(video=np.array([image]), video2=colors, vector=flow, vector_step=10,
                                         video2_alpha=0.5, scale=0.5)
        rgba = flowanim.to_rgba()
        plt.imshow(rgba[0], interpolation='none')

    else:
        quiver_plot(out_flow, img=image, thres=0.1, show=True)


def quiver_plot(flow: np.ndarray, img: Optional[np.array] = None, filename: Optional[str] = None,
                thres: Optional[float] = None, show: bool = False) -> None:
    flow = _quiver_clean(flow, threshold=thres) if thres and thres > 1 else flow
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Setting up the quiver plot
    h, w = u.shape
    x = np.arange(0, w) + 0.5
    y = np.arange(0, h)[::-1] + 0.5
    xp, yp = np.meshgrid(x, y)

    # ploting the result
    fig, ax = plt.subplots()

    plot_title = filename if filename else "Endo motion estimator"
    ax.set_title(plot_title)

    mag = np.hypot(u, v)
    q = ax.quiver(xp, yp, u, v, mag, units='x', scale=0.1, alpha=0.1)
    ax.axis('equal')

    if img is not None:  # merge plot with image (if available)
        ax.imshow(img)  # , extent=[0, w, h, 0])

    qk = ax.quiverkey(q, 0.9, 0.9, 1, r'$1 \frac{pix}{s}$', labelpos='E', coordinates='figure')
    if show:
        bottom, top = plt.ylim()  # return the current ylim
        plt.ylim((top, bottom))  # set the ylim to bottom, top
        plt.show()

    if filename is not None:
        assert type(filename) is str, "File is not str (%r)" % str(filename)
        assert filename[-4:] == '.png', "File extension is not an image format (%r)" % filename[-4:]
        plt.savefig(filename)

    plt.clf()


def _quiver_clean(flow: np.array, threshold: float = 0.05) -> np.array:
    flowval = np.linalg.norm(flow, axis=2)
    flow[flowval <= threshold] = np.nan  # avoid plotting vector

    return flow


def _checkstat(data):
    sns.distplot(data, hist=True, kde=True,
                 bins=int(180 / 5), color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})
    plt.show()


def get_label(flopaths: List[str,], labeldir: str = './labels'):
    labels = {}

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


def _velo_mean(flo: np.array, mask: Optional[np.array] = None):
    if mask is None:
        flo_mag = np.linalg.norm(flo, axis=-1)
        flo_mag_clean = flo_mag[~np.isnan(flo_mag)]

    else:
        flo_clean = flo[mask]
        flo_mag_clean = np.linalg.norm(flo_clean, axis=-1)

    return np.mean(flo_mag_clean)


if __name__ == '__main__':
    # Use flowviz (uncomment for usage)
    flodir = "./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif-0_end"
    imdir = "./frames/Test 03 L3 NAOCL 22000 fpstif"
    use_flowviz(flodir, imdir, start_at=4900, num_images=50, lossless=False)

    # Use get_label (uncomment for usage)
    # flopaths = ["./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif-4900_50/Test 03 L3 NAOCL 22000 fpstif_04900_out.flo"]
    # labels = get_label(flopaths)
    # v1 = labels[flopaths[0]]['v1']
    # v1_mean = _velo_mean(v1['flo'], mask=v1['mask'])

    # Image source
    imdir = "./frames"
    inputdir = 'Test 03 L3 NAOCL 22000 fpstif-4900_30'

    # Flow source
    flodirs = [os.path.join('./results', netname) for netname in ['Hui-LiteFlowNet', 'PIV-LiteFlowNet-en']]
    flosubdirs = [inputdir + f"_{i}" if i else inputdir for i in [None, 5, 10, 100]]

    floname = os.path.join(flodirs[0], flosubdirs[0], 'Test 03 L3 NAOCL 22000 fpstif_04900_out.flo')

    # Plotting
    out_flow = read_flow(floname)
    flowstat = np.linalg.norm(out_flow, axis=2).flatten()
    pt1, pt2 = [8, 120], [22, 136]
    slice_flow = out_flow[pt1[0]:pt2[0], pt1[1]:pt2[1]]
    slice_mag = np.hypot(slice_flow[:, :, 0], slice_flow[:, :, 1])
    slice_mag_mean = np.mean(slice_mag)
    # _checkstat(flowstat)

    # quiver_plot(out_flow, thres=0.5, show=True)
    singleplot(floname, imdir, use_flowviz=True)

    print('DONE!')
