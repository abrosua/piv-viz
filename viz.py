import os
import sys
import imageio
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from flowviz import colorflow, animate

import utils  # Import the utils directory as module

import matplotlib
matplotlib.use('TkAgg')


def color_plot(resolution: int = 256, vector_step: int = 1) -> None:
    """
    Plotting the color code
    """
    # Color plot
    x_tmp, y_tmp = np.meshgrid(np.arange(resolution)[::-1], np.arange(resolution)[::-1])
    flo_tmp = np.dstack([x_tmp, y_tmp]) - (resolution - 1) / 2
    # flo_tmp[:, :, 1] *= -1
    flo_tmp_color = colorflow.motion_to_color(flo_tmp)

    xk, yk = x_tmp[::vector_step, ::vector_step], y_tmp[::vector_step, ::vector_step]
    uk, vk = flo_tmp[::vector_step, ::vector_step, 0], flo_tmp[::vector_step, ::vector_step, 1]

    plt.figure()
    plt.imshow(flo_tmp_color)
    plt.quiver(xk, yk, uk, -vk)
    plt.show()


def clean_plot(flopaths: List[str], vector_step: int = 1, savefig: bool = False, show: bool = False) -> None:
    """
    Visualizing the image and the vector field within the defined label.
    params:
        flopaths: List of the flow vector (.flo) file paths.
        vector_step: Number of step to skip for the quiver vector visualization.
        savefig: Boolean variable for saving the plot or not
        show: Boolean variable for displaying the plot or not
    """
    labels = utils.get_label(flopaths)

    for flopath, label in labels.items():
        bname = os.path.basename(flopath).rsplit('_', 1)[0]

        # Image masking
        imdir_base = os.path.basename(os.path.dirname(flopath)).rsplit('-', 1)[0]
        impath = os.path.join("./frames", imdir_base, bname + ".tif")
        masked_img = imageio.imread(impath)  # Read the raw image

        # Add flow field
        flow = label['flow']
        flow_vec = flow['flo']
        u, v = flow_vec[:, :, 0], flow_vec[:, :, 1]

        flow_vec[~flow['mask']] = 0.0  # Replacing NaNs with zero to convert the value into RGB
        flo_color = colorflow.motion_to_color(flow_vec)

        # Merging image and plot the result
        masked_img[flow['mask']] = 0
        flo_color[~flow['mask']] = 0
        merge_img = masked_img + flo_color  # Superpose the image and flow color visualization

        # Quiver plot config.
        h, w = u.shape
        x, y = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        xp, yp = x[::vector_step, ::vector_step], y[::vector_step, ::vector_step]
        up, vp = u[::vector_step, ::vector_step], v[::vector_step, ::vector_step]

        # fig = Figure(figsize=np.array((w, h)))
        # FigureCanvasAgg(fig)
        # ax = fig.add_axes((0, 0, 1, 1))

        # Viz
        plt.figure()
        plt.imshow(merge_img)
        plt.quiver(xp, yp, up, -vp)

        # fig.canvas.draw()
        if show:
            plt.show()

        if savefig:
            # Setting up the path name
            plotdir = os.path.join(os.path.dirname(os.path.dirname(flopath)), "viz")
            os.makedirs(plotdir) if not os.path.isdir(plotdir) else None
            # Saving the plot
            plotpath = os.path.join(plotdir, bname + "_viz.png")
            plt.savefig(plotpath, dpi=300, bbox_inches='tight')

        plt.clf()


if __name__ == "__main__":
    # INPUT
    flopaths = ["./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif-0_end/Test 03 L3 NAOCL 22000 fpstif_04900_out.flo"]
    clean_plot(flopaths, vector_step=4, savefig=True)
    # color_plot(vector_step=4)

    print("DONE!")
