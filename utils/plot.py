import sys
import os
import imageio
from typing import Optional, List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
                                         video2_alpha=0.5)
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


def checkstat(data):
    sns.distplot(data, hist=True, kde=True,
                 bins=int(180 / 5), color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})
    plt.show()


if __name__ == "__main__":
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
    # checkstat(flowstat)

    # quiver_plot(out_flow, thres=0.5, show=True)
    singleplot(floname, imdir, use_flowviz=True)

    print("DONE!")
