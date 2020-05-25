import os
import imageio
from typing import Optional, List, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from flowviz import colorflow, animate

import utils
import matplotlib
matplotlib.use('TkAgg')


class FlowViz:
    def __init__(self, labelpaths: List[str], maxmotion: Optional[float] = None,
                 file_extension: Optional[str] = None, show: bool = False,
                 verbose: int = 0) -> None:
        """
        Flow visualization instance
        params:
            flopaths:
            netname:
            file_extension:
            show:
            verbose:
        """
        self.labelpaths = labelpaths
        self.maxmotion = maxmotion
        self.file_extension = file_extension
        self.show = show
        self.verbose = verbose

    def __call__(self, netname: str, vector_step: int = 1, use_color: bool = True, use_quiver: bool = True,
                 vorticity: bool = False, crop_window: Union[int, Tuple[int, int, int, int]] = 0,
                 **quiver_key) -> None:
        """
        Main
        params:
            vector_step:
            use_color:
            use_quiver:
            vorticity:
            crop_window:
        """
        for labelpath in self.labelpaths:
            # Init.
            label = utils.Label(labelpath, netname, verbose=self.verbose)
            bname = os.path.splitext(os.path.basename(labelpath))[0]
            keyname = ''

            # Add flow field
            flow, mask = label.get_flo("flow")
            if flow is None or mask is None:  # Skipping label file that doesn't have the flow label!
                continue
            # h, w, c = flow.shape  # Create image cropper
            # crop_window = (crop_window,) * 4 if type(crop_window) is int else crop_window
            # assert len(crop_window) == 4

            # Cropping the flow
            flow_crop = utils.array_cropper(flow, crop_window)
            mask_crop = utils.array_cropper(mask, crop_window)
            u, v = flow_crop[:, :, 0], flow_crop[:, :, 1]

            flow_crop[~mask_crop] = 0.0  # Replacing NaNs with zero to convert the value into RGB
            if vorticity:
                flo_color = None  #TODO create vorticity calculation function!
            else:
                flo_color = colorflow.motion_to_color(flow_crop, maxmotion=self.maxmotion)
            flo_color[~mask_crop] = 0  # Merging image and plot the result

            # Image masking
            impath = label.img_path
            img = imageio.imread(impath)  # Read the raw image
            masked_img = utils.array_cropper(img, crop_window)

            if use_color:
                # Superpose the image and flow color visualization if use_color is activated
                masked_img[mask_crop] = 0
                merge_img = masked_img + flo_color
                keyname += 'c'
            else:
                masked_img[mask_crop] = 255
                merge_img = masked_img

            # Viz
            plt.figure()
            plt.imshow(merge_img)
            if use_quiver:
                self.quiver_plot(u, v, mask_crop, vector_step, vector_color = not use_color, **quiver_key)
                keyname += 'q'
            # Erasing the axis number
            plt.xticks([])
            plt.yticks([])

            if self.show:
                plt.show()

            if self.file_extension:
                # Setting up the path name
                plotdir = os.path.join("./results", netname, label.img_dir, "viz")
                os.makedirs(plotdir) if not os.path.isdir(plotdir) else None
                # Saving the plot
                plotpath = os.path.join(plotdir, bname + f"_{keyname}viz.{self.file_extension}")
                plt.savefig(plotpath, dpi=300, bbox_inches='tight')

            plt.clf()

    @staticmethod
    def quiver_plot(u, v, mask, vector_step: int = 1, vector_color: bool = False,
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


class FlowVideo:
    def __init__(self, flodir: str, start_at: int = 0, num_images: int = -1, lossless: bool = True,
                 crop_window: Union[int, Tuple[int, int, int, int]] = 0):
        """
        Generating flow video.
        """
        # Obtaining the flo files and file basename
        self.flows, self.flonames = utils.read_flow_collection(flodir, start_at=start_at, num_images=num_images,
                                                               crop_window=crop_window)
        fname_addition = f"-{start_at}_all" if num_images < 0 else f"-{start_at}_{num_images}"
        name_list = os.path.normpath(flodir).split(os.sep)
        self.imdir, self.netname = name_list[-2], name_list[-3]
        self.fname = self.imdir + fname_addition

        # Saving the video
        self.viddir = os.path.join(os.path.dirname(flodir), "videos")
        if not os.path.isdir(self.viddir):
            os.makedirs(self.viddir)

        self.lossless = lossless
        self.crop_window = crop_window

    def use_flowviz(self):
        """
        Visualization using flowviz module
        """
        print("Optical flow visualization using flowviz by marximus...")

        # Manage the input images
        video_list = []
        for floname in self.flonames:
            bname = os.path.basename(floname).rsplit('_', 1)[0]
            video_list.append(get_image(basename=bname, imdir=self.imdir, crop_window=self.crop_window))
        video = np.array(video_list)

        # Previewing the files
        print('Video shape: {}'.format(video.shape))
        print('Flows shape: {}'.format(self.flows.shape))

        # Define the output files
        colors = colorflow.motion_to_color(self.flows)
        flowanim = animate.FlowAnimation(video=video, video2=colors, vector=self.flows, vector_step=10,
                                      video2_alpha=0.5)

        if self.lossless:
            vidpath = os.path.join(self.viddir, self.fname + ".avi")
            flowanim.save(vidpath, codec="ffv1")
        else:
            vidpath = os.path.join(self.viddir, self.fname + ".mp4")
            flowanim.save(vidpath)
        print(f"Finish saving the video file ({vidpath})!")

    def use_manual(self, labelpath: str, verbose: int = 0):
        """
        Visualization using in-house function
        """
        print("Optical flow visualization by abrosua...")

        if os.path.isfile(labelpath):  # Using single label file.
            label_main = utils.Label(labelpath, netname=self.netname, verbose=verbose)
        elif os.path.isdir(labelpath):  # Using multiple label files.
            label_main = None
        else:
            raise ValueError(f"Label path is NOT found at '{labelpath}'!")

    def _grab_frame(self):
        pass


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
    # Color plot
    pts = np.linspace(-maxmotion, maxmotion, num=resolution)
    x, y = np.meshgrid(pts, pts)
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
    # Generate colormap
    color_map(resolution=256, maxmotion=4, show=True)

    print("DONE!")
