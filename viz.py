import os
import sys
import imageio
from typing import Optional, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from flowviz import colorflow, animate

import utils  # Import the utils directory as module

import matplotlib
matplotlib.use('TkAgg')


class FlowViz:
    def __init__(self, flopaths: List[str], file_extension: Optional[str] = None, show: bool = False,
                 verbose: int = 0) -> None:
        """
        Flow visualization instance
        """
        self.labels = utils.get_label(flopaths, verbose=verbose)
        self.file_extension = file_extension
        self.show = show

    def __call__(self, vector_step: int = 1, use_color: bool = True, use_quiver: bool = True, vorticity: bool = False,
                 crop_window: Union[int, Tuple[int, int, int, int]] = 0, **quiver_key) -> None:
        """
        Main
        params:
            method:
        """
        for flopath, label in self.labels.items():
            bname = os.path.basename(flopath).rsplit('_', 1)[0]

            # Image masking
            imdir_base = os.path.basename(os.path.dirname(flopath)).rsplit('-', 1)[0]
            impath = os.path.join("./frames", imdir_base, bname + ".tif")
            img = imageio.imread(impath)  # Read the raw image

            # Create image cropper
            h, w, c = img.shape
            crop_window = (crop_window,) * 4 if type(crop_window) is int else crop_window
            assert len(crop_window) == 4
            masked_img = img[crop_window[0] : h-crop_window[1], crop_window[2] : w-crop_window[2], :]

            # Add flow field
            flow = label['flow']
            flow_vec = flow['flo'][crop_window[0] : h-crop_window[1], crop_window[2] : w-crop_window[3], :]
            flow_mask = flow['mask'][crop_window[0] : h-crop_window[1], crop_window[2] : w-crop_window[3]]
            u, v = flow_vec[:, :, 0], flow_vec[:, :, 1]

            flow_vec[~flow_mask] = 0.0  # Replacing NaNs with zero to convert the value into RGB
            if vorticity:
                flo_color = None  #TODO create vorticity calculation function!
            else:
                flo_color = colorflow.motion_to_color(flow_vec)

            # Merging image and plot the result
            flo_color[~flow_mask] = 0

            if use_color:
                # Superpose the image and flow color visualization if use_color is activated
                masked_img[flow_mask] = 0
                merge_img = masked_img + flo_color
            else:
                masked_img[flow_mask] = 255
                merge_img = masked_img

            # Viz
            plt.figure()
            plt.imshow(merge_img)
            self.quiver_plot(u, v, vector_step, vector_color = not use_color, **quiver_key)
            # Erasing the axis number
            plt.xticks([])
            plt.yticks([])

            if self.show:
                plt.show()

            if self.file_extension:
                # Setting up the path name
                plotdir = os.path.join(os.path.dirname(os.path.dirname(flopath)), "viz")
                os.makedirs(plotdir) if not os.path.isdir(plotdir) else None
                # Saving the plot
                plotpath = os.path.join(plotdir, bname + f"_viz.{self.file_extension}")
                plt.savefig(plotpath, dpi=300, bbox_inches='tight')

            plt.clf()

    def quiver_plot(self, u, v, vector_step: int = 1, vector_color: bool = False, **quiver_key):
        """
        Quiver plot config.
        params:
            u: Flow displacement at x-direction.
            v: Flow displacement at y-direction.
            vector_step: Number of step for displaying the vector
        """
        h, w = u.shape
        x, y = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        xp, yp = x[::vector_step, ::vector_step], y[::vector_step, ::vector_step]
        up, vp = u[::vector_step, ::vector_step], v[::vector_step, ::vector_step]
        mag = np.hypot(up, vp)

        if vector_color:
            # Setting the vector color
            colormap = cm.inferno
            q = plt.quiver(xp, yp, up, -vp, mag, cmap=colormap, units='xy', width=0.005*w)
            plt.colorbar()
        else:
            q = plt.quiver(xp, yp, up, -vp, units='xy', width=0.004*w)

        qk = plt.quiverkey(q, **quiver_key) if quiver_key else None

    @staticmethod
    def color_map(resolution: int = 256, vector_step: int = 1,
                  show: bool = False, filename: Optional[str] = None) -> None:
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

        plt.show() if show else None
        plt.savefig(filename, dpi=300, bbox_inches='tight') if filename else None
        plt.clf()


if __name__ == "__main__":
    # INPUT
    ext = 'eps'  # 'png' for standard image and 'eps' for latex format
    show_figure = False
    flopaths = ["./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif-0_end/Test 03 L3 NAOCL 22000 fpstif_04900_out.flo"]

    # Main
    cropper = (100, 0, 0, 0)
    flow_visualizer = FlowViz(flopaths, file_extension=ext, show=show_figure, verbose=1)
    flow_visualizer(vector_step=6, use_color=False, crop_window=cropper)

    print("DONE!")
