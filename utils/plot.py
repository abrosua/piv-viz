import os
import imageio
import math
from typing import Optional, List, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib import animation
from flowviz import colorflow, animate

import utils
import matplotlib
matplotlib.use('TkAgg')


class FlowViz:
    def __init__(self, labelpaths: List[str], netname: str, maxmotion: Optional[float] = None, vector_step: int = 1,
                 use_color: bool = True, use_quiver: bool = True, calc_vorticity: bool = False,
                 crop_window: Union[int, Tuple[int, int, int, int]] = 0, verbose: int = 0) -> None:
        """
        Flow visualization instance
        params:
            flopaths:
            netname:
            file_extension:
            show:
            verbose:
        """
        assert len(labelpaths) > 0  # Input sanity checking
        self.labelpaths = labelpaths
        self.netname = netname

        # Logic gate
        self.use_color = use_color
        self.use_quiver = use_quiver
        self.vorticity = calc_vorticity
        self.verbose = verbose

        # Variables
        self.maxmotion = maxmotion
        self.vector_step = vector_step
        self.crop_window = crop_window

        # Init.
        self.img_dir = ""
        self.keyname = ""
        if self.use_color:
            self.keyname += "c"
        if self.use_quiver:
            self.keyname += "q"
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def plot(self, file_extension: Optional[str] = None, show: bool = False, savedir: Optional[str] = None):
        """
        Plotting
        """
        for labelpath in self.labelpaths:
            label = utils.Label(labelpath, self.netname, verbose=self.verbose)
            bname = os.path.splitext(os.path.basename(labelpath))[0]

            check_flow = label.get_flo("flow")
            if check_flow[0] is None or check_flow[1] is None:
                continue
            else:
                self.draw_frame(label)

            if show:
                plt.show()

            if file_extension:
                # Setting up the path name
                plotdir = os.path.join("./results", self.netname, self.img_dir, "viz") if savedir is None else savedir
                os.makedirs(plotdir) if not os.path.isdir(plotdir) else None
                # Saving the plot
                plotpath = os.path.join(plotdir, bname + f"_{self.keyname}viz.{file_extension}")
                plt.savefig(plotpath, dpi=300, bbox_inches='tight')

            plt.clf()

    def video(self, flodir: str, start_at: int = 0, num_images: int = -1,
              fps: int = 30, dpi: int = 300, lossless: bool = True):
        """
        Generating video
        """
        # Flow init.
        flows, flonames = utils.read_flow_collection(flodir, start_at=start_at, num_images=num_images,
                                                     crop_window=self.crop_window)
        fname_addition = f"-{start_at}_all" if num_images < 0 else f"-{start_at}_{num_images}"
        name_list = os.path.normpath(flodir).split(os.sep)
        self.img_dir = str(name_list[-2])

        # Label init.
        if len(self.labelpaths) == 1:  # Single label mode
            label_main = utils.Label(self.labelpaths[0], netname=self.netname, verbose=self.verbose)
        elif len(self.labelpaths) >= len(flonames):  # Multiple labels mode
            label_main = None
        else:  # Raising ERROR
            raise ValueError("Multiple labels mode is used, but the number of input labelpaths is NOT sufficient!")

        # Video writer config.
        vidname = self.img_dir + fname_addition + f"_{self.keyname}vid"
        if lossless:
            vidpath = os.path.join(os.path.dirname(flodir), "videos", vidname + ".avi")
            writer = animation.FFMpegFileWriter(fps=fps, bitrate=-1, codec="ffv1")
        else:
            vidpath = os.path.join(os.path.dirname(flodir), "videos", vidname + ".mp4")
            writer = animation.FFMpegFileWriter(fps=fps, bitrate=-1)

        with writer.saving(self.fig, vidpath, dpi=dpi):  # TODO: Fix the writer! And test it later on!
            id_flo, id_label = 0, 0

            while id_flo < len(flonames):
                floname = flonames[id_flo]
                flonum = int(os.path.basename(floname).split("_")[-2])

                if label_main is None:  # Multiple labels mode
                    labelpath = self.labelpaths[id_label]
                    labelnum = int(os.path.splitext(os.path.basename(labelpath))[0].rsplit("_", 1)[-1])
                    id_label += 1

                    if labelnum < flonum:
                        continue
                    elif labelnum > flonum:
                        raise ValueError(f"Label file is NOT found for flow file at '{floname}'")
                    else:
                        label = utils.Label(labelpath, netname=self.netname, verbose=self.verbose)

                    check_flow = label.get_flo("flow")
                    if None in check_flow:  # Checking the flow label availability
                        raise ValueError(f"Flow label is NOT found in '{labelpath}'")
                else:  # Single label mode
                    label = label_main

                # Gathering each frame
                id_flo += 1
                self.draw_frame(label)
                writer.grab_frame(bbox_inches='tight')

    def draw_frame(self, label):
        """
        Drawing each single frame.
        """
        # Add flow field
        flow, mask = label.get_flo("flow")
        if flow is None or mask is None:  # Skipping label file that doesn't have the flow label!
            return None

        # Cropping the flow
        flow_crop = utils.array_cropper(flow, self.crop_window)
        mask_crop = utils.array_cropper(mask, self.crop_window)
        u, v = flow_crop[:, :, 0], flow_crop[:, :, 1]

        flow_crop[~mask_crop] = 0.0  # Replacing NaNs with zero to convert the value into RGB
        if self.vorticity:
            flo_color = None  # TODO create vorticity calculation function!
        else:
            flo_color = colorflow.motion_to_color(flow_crop, maxmotion=self.maxmotion)
        flo_color[~mask_crop] = 0  # Merging image and plot the result

        # Image masking
        impath = label.img_path
        img = imageio.imread(impath)  # Read the raw image
        masked_img = utils.array_cropper(img, self.crop_window)

        if self.use_color:
            # Superpose the image and flow color visualization if use_color is activated
            masked_img[mask_crop] = 0
            merge_img = masked_img + flo_color
        else:
            masked_img[mask_crop] = 255
            merge_img = masked_img

        # Viz
        self.ax.imshow(merge_img)
        if self.use_quiver:
            self.quiver_plot(self.ax, u, v, mask_crop, self.vector_step, vector_color=not self.use_color)
        # Erasing the axis number
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    @staticmethod
    def quiver_plot(ax, u, v, mask, vector_step: int = 1, vector_color: bool = False,
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
            q = ax.quiver(X, Y, U, -V, MAG, cmap=colormap,
                          units='xy', width=width_factor*w)
            ax.colorbar()
        else:
            q = ax.quiver(X, Y, U, -V,
                          units='xy', width=width_factor*w)

        qk = ax.quiverkey(q, **quiver_key) if quiver_key else None


def get_image(basename, imdir, crop_window: Union[int, Tuple[int, int, int, int]] = 0) -> np.array:
    imname = basename + ".tif"
    impath = os.path.join(imdir, imname)
    img = imageio.imread(impath)

    return utils.array_cropper(img, crop_window)


def vid_flowviz(flodir, imdir, start_at: int = 0, num_images: int = -1, lossless: bool = True):
    """
    Visualization using flowviz module
    """
    print("Optical flow visualization using flowviz by marximus...")

    # Obtaining the flo files and file basename
    flows, flonames = utils.read_flow_collection(flodir, start_at=start_at, num_images=num_images)
    fname_addition = f"-{start_at}_all" if num_images < 0 else f"-{start_at}_{num_images}"
    fname = os.path.basename(os.path.dirname(flodir)) + fname_addition

    # Manage the input images
    video_list = []
    for floname in flonames:
        filename = str(os.path.basename(floname).rsplit('_', 1)[0]) + ".tif"
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
                                     video2_alpha=0.5)

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


def color_map(resolution: int = 256, maxmotion: float = 1.0, show: bool = False, filename: Optional[str] = None
              ) -> None:
    """
    Plotting the color code
    params:
        resolution: Colormap image resolution.
        maxmotion: Maximum flow motion.
        show: Option to display the plot or not.
        filename: Full file path to save the plot; use None for not saving the plot!
    """
    # Init.
    bname = os.path.basename(filename).rsplit("_", 1)[0]

    # Color plot
    pts = np.linspace(-maxmotion, maxmotion, num=resolution)
    x, y = np.meshgrid(pts, pts)
    flo = np.dstack([x, y])
    flo_color = colorflow.motion_to_color(flo)

    # Plotting the image
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(flo_color, extent=[-math.ceil(maxmotion), math.ceil(maxmotion),
                                 -math.ceil(maxmotion), math.ceil(maxmotion)])
    ax.set_title(f"{bname} Colormap")

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
    # Generate colormap
    color_map(resolution=256, maxmotion=4, show=True)

    print("DONE!")
