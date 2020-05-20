import os
import sys
import imageio
from typing import Optional, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from flowviz import colorflow

import utils  # Import the utils directory as module

import matplotlib
matplotlib.use('TkAgg')


class FlowViz:
    def __init__(self, flopaths: List[str], file_extension: Optional[str] = None, show: bool = False,
                 verbose: int = 0) -> None:
        """
        Flow visualization instance
        params:
            flopaths:
            file_extension:
            show:
            verbose:
        """
        self.labels = utils.get_label(flopaths, verbose=verbose)
        self.file_extension = file_extension
        self.show = show

    def __call__(self, vector_step: int = 1, use_color: bool = True, use_quiver: bool = True, vorticity: bool = False,
                 crop_window: Union[int, Tuple[int, int, int, int]] = 0, **quiver_key) -> None:
        """
        Main
        params:
            vector_step:
            use_color:
            use_quiver:
            vorticity:
            crop_window:
        """
        for flopath, label in self.labels.items():
            # Init.
            bname = os.path.basename(flopath).rsplit('_', 1)[0]
            keyname = ''

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
                keyname += 'c'
            else:
                masked_img[flow_mask] = 255
                merge_img = masked_img

            # Viz
            plt.figure()
            plt.imshow(merge_img)
            if use_quiver:
                self.quiver_plot(u, v, flow_mask, vector_step, vector_color = not use_color, **quiver_key)
                keyname += 'q'
            # Erasing the axis number
            plt.xticks([])
            plt.yticks([])

            if self.show:
                plt.show()

            if self.file_extension:
                # Setting up the path name
                plotdir = os.path.join(os.path.dirname(os.path.dirname(flopath)), "viz", imdir_base)
                os.makedirs(plotdir) if not os.path.isdir(plotdir) else None
                # Saving the plot
                plotpath = os.path.join(plotdir, bname + f"_{keyname}viz.{self.file_extension}")
                plt.savefig(plotpath, dpi=300, bbox_inches='tight')

            plt.clf()

    def quiver_plot(self, u, v, mask, vector_step: int = 1, vector_color: bool = False,
                    **quiver_key) -> None:
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

        # Plotting preparation and Masking the data
        maskp = mask[::vector_step, ::vector_step]
        X, Y, U, V, MAG = xp[maskp], yp[maskp], up[maskp], vp[maskp], mag[maskp]
        width_factor = 0.004

        if vector_color:
            # Setting the vector color
            colormap = cm.inferno
            q = plt.quiver(X, Y, U, -V, MAG, cmap=colormap,
                           units='xy', width=width_factor*w)
            plt.colorbar()
        else:
            q = plt.quiver(X, Y, U, -V,
                           units='xy', width=width_factor*w)

        qk = plt.quiverkey(q, **quiver_key) if quiver_key else None

    @staticmethod
    def color_map(resolution: int = 256, maxmotion: float = 1.0, vector_step: int = 1,
                  show: bool = False, filename: Optional[str] = None) -> None:
        """
        Plotting the color code
        params:
            resolution: Colormap image resolution.
            maxmotion: Maximum flow motion.
            vector_step: Vector shifting variable
            show: Option to display the plot or not.
            filename: Full file path to save the plot; use None for not saving the plot!
        """
        # Color plot
        pts = np.linspace(-maxmotion, maxmotion, num=resolution)
        x, y = np.meshgrid(pts/maxmotion, pts/maxmotion)
        flo = np.dstack([x, y])
        flo_color = colorflow.motion_to_color(flo)

        # Plotting the image
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(flo_color, extent=[-maxmotion, maxmotion, -maxmotion, maxmotion])

        # Erasing the zeros
        func = lambda x, pos: "" if np.isclose(x, 0) else x
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(func))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(func))

        # Putting the axis in the middle of the graph, passing through (0,0)
        ax.spines['left'].set_position('center')  # Move left y-axis and bottim x-axis to centre
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')  # Eliminate upper and right axes
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')  # Show ticks in the left and lower axes only
        ax.yaxis.set_ticks_position('left')

        plt.show() if show else None
        plt.savefig(filename, dpi=300, bbox_inches='tight') if filename else None
        plt.clf()


if __name__ == "__main__":
    # INPUT
    ext = None  # 'png' for standard image and 'eps' for latex format; use None to disable!
    show_figure = False
    flopaths = ["./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif-0_end/Test 03 L3 NAOCL 22000 fpstif_04900_out.flo"]

    # Main
    cropper = (100, 0, 0, 0)
    flow_visualizer = FlowViz(flopaths, file_extension=ext, show=show_figure, verbose=1)
    flow_visualizer(vector_step=5, use_quiver=True, use_color=True, crop_window=cropper)
    flow_visualizer.color_map(maxmotion=4, show=True)

    print("DONE!")
