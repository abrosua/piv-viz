import sys
import os
from typing import Optional

from PIL import Image
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
# matplotlib.use('Qt5Agg')

pivdir = "../thesis_faber"
sys.path.append(pivdir)
from src.utils_plot import read_flow, motion_to_color


def singleplot(floname, imdir):
    # Generate image name from the flo file
    basename = str(os.path.basename(floname).rsplit('_', 1)[0])
    subdir = os.path.basename(os.path.dirname(floname))
    imname = os.path.join(imdir, subdir, basename + ".tif")

    # plotting image
    im = Image.open(imname).convert('RGB')
    plt.imshow(im)
    out_flow = read_flow(floname)
    quiver_plot(out_flow, thres=0.1, show=True)


def quiver_plot(flow: np.ndarray, filename: Optional[str] = None, thres: Optional[float] = None,
                show: bool = False) -> None:
    flow = _quiver_clean(flow, threshold=thres) if thres and thres > 1 else flow
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Setting up the quiver plot
    h, w = u.shape
    x = np.arange(0, w) + 0.5
    y = np.arange(0, h)[::-1] + 0.5
    xp, yp = np.meshgrid(x, y)

    # ploting the result
    fig1, ax1 = plt.subplots()
    plot_title = filename if filename else "Endo motion estimator"
    ax1.set_title(plot_title)

    mag = np.hypot(u, v)
    q = ax1.quiver(xp, yp, u, v, mag, units='x', scale=0.1)
    ax1.axis('equal')
    qk = ax1.quiverkey(q, 0.9, 0.9, 1, r'$1 \frac{pix}{s}$', labelpos='E', coordinates='figure')
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


if __name__ == '__main__':
    # Image source
    imdir = "./frames"

    # Flow source
    inputdir = 'Test 03 L3 NAOCL 22000 fpstif'
    flodirs = [os.path.join('./results', netname) for netname in ['Hui-LiteFlowNet', 'PIV-LiteFlowNet-en']]
    flosubdirs = [inputdir + f"_{i}" if i else inputdir for i in [None, 5, 10, 100]]

    floname = os.path.join(flodirs[0], flosubdirs[0], 'Test 03 L3 NAOCL 22000 fpstif_0000_out.flo')

    # Plotting
    out_flow = read_flow(floname)
    flowstat = np.linalg.norm(out_flow, axis=2).flatten()
    # _checkstat(flowstat)

    # quiver_plot(out_flow, thres=0.5, show=True)
    singleplot(floname, imdir)

    print('DONE!')
