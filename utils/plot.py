import os
import imageio
import math
from glob import  glob
from typing import Optional, List, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib import animation
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from flowviz import colorflow, animate
from PIL import Image
from tqdm import tqdm

import utils
import matplotlib
matplotlib.use('TkAgg')


class FlowViz:
    def __init__(self, labelpaths: List[str], flodir: str, maxmotion: Optional[float] = None, vector_step: int = 1,
                 use_color: int = 0, use_quiver: bool = False, color_type: Optional[str] = None, key: str = "flow",
                 crop_window: Union[int, Tuple[int, int, int, int]] = 0, calib: float = 1.0, fps: int = 1,
                 post_factor: Optional[float] = None, use_stereo: bool = False, verbose: int = 0) -> None:
        """
        Flow visualization instance
        params:
            flopaths:
            netname:
            file_extension:
            show:
            verbose:
        """
        # Input sanity checking
        assert len(labelpaths) > 0
        assert type(use_color) == int and use_color < 3
        if color_type is not None:
            assert color_type in ["vort", "shear", "normal", "mag"]

        # # -----> Setting up the directory
        self.labelpaths = labelpaths
        self.flodir = flodir
        self.workdir = utils.split_abspath(flodir, cutting_path="flow")[0]
        self.img_bname = os.path.basename(self.workdir)

        # Logic gate
        self.use_color = use_color
        self.use_quiver = use_quiver
        self.use_stereo = use_stereo
        self.color_type = color_type
        self.verbose = verbose

        # Variables
        self.maxmotion = maxmotion
        self.vector_step = vector_step
        self.crop_window = crop_window
        self.fps, self.calib, self.post_factor = fps, calib, post_factor
        self.velocity_factor = calib * fps
        self.key = key

        # Init.
        self.fig = plt.figure()
        self.is_init = False

        if use_stereo:
            self.ax = self.fig.add_subplot(1, 2, 1)
            self.ax_stereo = self.fig.add_subplot(1, 2, 2, projection='3d')
            self.ax_stereo.set_xlabel('X')
            self.ax_stereo.set_ylabel('Y')
            self.ax_stereo.set_zlabel('Z')
            self.ax_stereo.set_zlim3d(0, maxmotion)
        else:
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.set_xticks([])  # Erasing the axis number
            self.ax.set_yticks([])

        plt.subplots_adjust(right=0.85) if color_type in ["mag", "vort", "normal", "shear"] else None

        self.im1, self.quiver = None, None
        self.keyname = ""

        # Scalar mapping
        if color_type in ["vort", "shear", "normal"]:
            self.norm = colors.TwoSlopeNorm(vcenter=0.0) if self.maxmotion is None else \
                colors.TwoSlopeNorm(vmin=-self.maxmotion, vcenter=0.0, vmax=self.maxmotion)
            self.scalar_map = cm.ScalarMappable(norm=self.norm, cmap=cm.coolwarm)
        else:
            self.norm = colors.Normalize() if self.maxmotion is None else \
                colors.Normalize(vmin=0, vmax=self.maxmotion)
            self.scalar_map = cm.ScalarMappable(norm=self.norm, cmap=cm.hot_r)

        if (use_quiver and not use_color) or color_type in ["mag", "vort", "shear", "normal"]:
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = self.fig.colorbar(self.scalar_map, cax=cax)

            if color_type == "vort":
                unit = "1/ms"
            elif color_type in ["shear", "normal"]:
                unit = "N/m2"
            else:
                unit = "pix" if self.velocity_factor == 1.0 else "mm/s"
            clb.ax.set_title(unit)

        if color_type is not None:
            self.keyname += f"{color_type}-"

        if self.use_color == 1:
            self.keyname += "c"
        elif self.use_color == 2:
            self.keyname += "t"
            self.im2 = None
        if self.use_quiver:
            self.keyname += "q"
        if self.use_stereo:
            self.keyname += "3"

    def plot(self, ext: Optional[str] = None, show: bool = False, savedir: Optional[str] = None,
             **quiver_key) -> None:
        """
        Plotting
        """
        init_plot = True

        for labelpath in self.labelpaths:
            label = utils.Label(labelpath, self.flodir, verbose=self.verbose)
            bname = os.path.splitext(os.path.basename(labelpath))[0]

            # Add flow field
            flow, mask = label.get_flo(self.key, use_stereo=self.use_stereo)
            if flow is None or mask is None:  # Skipping label file that doesn't have the flow label!
                continue
            else:
                flow = flow * self.velocity_factor

            # Reading flow and image files
            flow_crop = utils.array_cropper(flow, self.crop_window)
            mask_crop = utils.array_cropper(mask, self.crop_window)
            img = np.array(Image.open(label.img_path).convert("RGB"))
            # img = imageio.imread(label.img_path)  # Read the raw image
            img_crop = utils.array_cropper(img, self.crop_window)

            if init_plot:
                self._init_frame(img_crop, **quiver_key)  # Initialize the frame
                init_plot = False
            self._draw_frame(flow_crop, img_crop, mask_crop)  # Drawing each frame

            if show:
                plt.show()

            if ext:
                # -----> Setting up the path name
                plotdir = os.path.join(self.workdir, "viz") if savedir is None else savedir
                os.makedirs(plotdir) if not os.path.isdir(plotdir) else None
                # -----> Saving the plot
                plotpath = os.path.join(plotdir, bname + f"_{self.keyname}viz.{ext}")
                plt.savefig(plotpath, dpi=300, bbox_inches='tight')

            plt.clf()

    def multiplot(self, ext: Optional[str] = None, show: bool = False, start_at: int = 0, num_images: int = -1,
                  use_image: bool = True) -> None:
        """
        Generating multiple image plots with a single label.
        """
        # Flow init.
        assert num_images != 0
        flonames = sorted(glob(os.path.join(self.flodir, "*.flo")))
        flonames = flonames[start_at:] if num_images < 0 else flonames[start_at : num_images+start_at]
        num_images = len(flonames) if num_images < 0 else num_images

        # Frame looping init.
        if len(self.labelpaths) > 1:
            raise ValueError(f"Multiple labelpaths are found! ('{self.labelpaths}')")
        assert os.path.isfile(self.labelpaths[0])

        label = utils.Label(self.labelpaths[0], self.flodir, verbose=self.verbose)
        imgext = os.path.splitext(label.img_path)[-1]
        _, mask = label.get_flo(self.key, use_stereo=self.use_stereo)  # Create the mask
        mask_crop = utils.array_cropper(mask, self.crop_window)
        height, width = label.img_shape

        # Initiating directory naming
        imgdir = os.path.dirname(label.img_path)
        plotdir = os.path.join(self.workdir, f"plot-{start_at}_{num_images}")
        os.makedirs(plotdir) if not os.path.isdir(plotdir) else None

        for i, floname in tqdm(enumerate(flonames), desc=self.img_bname, unit="image", total=len(flonames)):
            bname = os.path.basename(floname).rsplit("_", 1)[0]
            if os.path.basename(imgdir).lower() in ["left", "right"]:
                bname_ext = f"-{os.path.basename(imgdir)[0].upper()}"
            else:
                bname_ext = ""

            # Importing the image
            imagepath = os.path.join(imgdir, bname + bname_ext + f"{imgext}")
            img = np.array(Image.open(imagepath).convert("RGB")) if use_image else np.zeros([height, width, 3], dtype=np.uint8)
            img_crop = utils.array_cropper(img, self.crop_window)

            # Add flow field
            flow_crop = utils.read_flow(floname, use_stereo=self.use_stereo, crop_window=self.crop_window
                                        ) * self.velocity_factor
            if not self.is_init:
                self._init_frame(img_crop)  # Initialize the frame
                self.is_init = True
            self._draw_frame(flow_crop, img_crop, mask_crop)  # Drawing each frame

            if show:
                plt.show()

            if ext:
                # Saving the plot
                plotpath = os.path.join(plotdir, bname + f"_{self.keyname}viz.{ext}")
                self.fig.savefig(plotpath, dpi=300, bbox_inches='tight')
                # plt.savefig(plotpath, dpi=300, bbox_inches='tight')

            # plt.clf()

    def video(self, ext: Optional[str] = None, start_at: int = 0, num_images: int = -1, fps: int = 1, dpi: int = 300
              ) -> None:
        """
        Generating video
        """
        # Flow init.
        flows, flonames = utils.read_flow_collection(self.flodir, start_at=start_at, num_images=num_images,
                                                     use_stereo=self.use_stereo, crop_window=self.crop_window)
        flows = flows * self.velocity_factor

        assert num_images != 0
        num_images = len(flonames) if num_images < 0 else num_images
        fname_addition = f"-{start_at}_{num_images}"

        # Frame looping init.
        labelpaths = []

        for i, floname in enumerate(flonames):
            if len(self.labelpaths) == 1:
                labelpath = self.labelpaths[0]
                labelpaths.append(labelpath)

            elif len(self.labelpaths) > 1:
                bname = os.path.basename(floname).rsplit("_", 1)[0]
                labeldir = os.path.dirname(self.labelpaths[0])
                labelpath = os.path.join(labeldir, bname + ".json")
                assert labelpath in self.labelpaths

            else:
                raise ValueError(f"Unknown labelpaths input! '{self.labelpaths}'")

            assert os.path.isfile(labelpath)

        # Initialize frame
        label = utils.Label(labelpaths[0], flodir=self.flodir, verbose=self.verbose)
        img = np.array(Image.open(label.img_path).convert("RGB"))
        img_tmp = utils.array_cropper(img, self.crop_window)
        self._init_frame(img_tmp)

        # Generate animation writer
        flowviz_anim = animation.FuncAnimation(self.fig, self._capture_frame, fargs=(labelpaths, flonames, flows),
                                               interval=1000/fps, frames=num_images)
        vidname = self.img_bname + fname_addition + f"_{self.keyname}vid"

        if ext is not None:
            vidpath = os.path.join(self.workdir, "videos", vidname + f".{ext}")
            if not os.path.isdir(os.path.dirname(vidpath)):
                os.makedirs(os.path.dirname(vidpath))

            if ext == "gif":  # GIF format
                writer = animation.writers["imagemagick"](fps=fps, metadata=dict(artist="abrosua"), bitrate=-1)
                flowviz_anim.save(vidpath, writer=writer)
            else:  # Video codec format
                writer = animation.writers["ffmpeg"](fps=fps, metadata=dict(artist="abrosua"), bitrate=-1)
                flowviz_anim.save(vidpath, writer=writer, dpi=dpi)

        self.ax.cla()

    def _capture_frame(self, idx, labelpaths: List[str], flonames: List[str], flows: np.array):
        """
        Iteration function for capturing vidoe frame.
        """
        labelpath = labelpaths[idx]
        floname = flonames[idx]

        # Instantiate the Label object
        label = utils.Label(labelpath, flodir=self.flodir, verbose=self.verbose)
        _, mask = label.get_flo(self.key, use_stereo=self.use_stereo)

        # Instantiate the image file path
        imgdir, imgname_tmp = os.path.split(label.img_path)
        bname = os.path.basename(floname).rsplit("_", 1)[0]

        if os.path.basename(os.path.dirname(floname)).lower() == "flow": # Check if the flo files are in Stereo format
            bname_ext = os.path.splitext(imgname_tmp)[-1]
        else:
            bname_ext = imgname_tmp.rsplit("-", 1)[-1]

        imgpath = os.path.join(imgdir, bname + bname_ext)

        # Gathering each frame
        img = np.array(Image.open(imgpath).convert("RGB"))
        img_crop = utils.array_cropper(img, self.crop_window)
        mask_crop = utils.array_cropper(mask, self.crop_window)

        self._draw_frame(flows[idx, :, :, :], img_crop, mask_crop)

    def _init_frame(self, image: np.array, **quiver_key):
        """
        Initializing frame for video writer.
        """
        # Image init.
        h, w, _ = image.shape
        self.im1 = self.ax.imshow(np.zeros_like(image))
        self.im2 = self.ax.imshow(np.zeros([h, w, 4], dtype=image.dtype)) if self.use_color == 2 else None

        # Quiver init.
        if self.use_quiver:
            x, y = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
            xp, yp = x[::self.vector_step, ::self.vector_step], y[::self.vector_step, ::self.vector_step]
            mag = np.hypot(xp, yp)
            width_factor = 0.004

            if not self.use_color:
                # Setting the vector color
                colormap = self.scalar_map.get_cmap()
                self.quiver = self.ax.quiver(xp, yp, np.zeros_like(xp), np.zeros_like(yp), np.zeros_like(mag),
                                             units='xy', width=width_factor*w, scale=width_factor*100, cmap=colormap)
            else:
                self.quiver = self.ax.quiver(xp, yp, np.zeros_like(xp), np.zeros_like(yp),
                                             units='xy', width=width_factor*w, scale=width_factor*100)

            qk = self.ax.quiverkey(self.quiver, **quiver_key) if quiver_key else None

            if self.use_stereo:  #TODO: Recheck the stereo plotting init.
                xs, ys = np.expand_dims(xp, axis=-1), np.expand_dims(yp, axis=-1)
                zs = np.zeros_like(xs)

                self.im_stereo = self.ax_stereo.plot_surface(x, y, np.zeros_like(x), linewidth=0, rstride=50,
                                                             cstride=50)
                self.quiver_stereo = self.ax_stereo.quiver(*(xs, ys, zs, zs, zs, zs), normalize=True) \
                    if self.use_color else \
                    self.ax_stereo.quiver(*(xs, ys, zs, zs, zs, zs), normalize=True, cmap=self.scalar_map.get_cmap())

    def _draw_frame(self, flow: np.array, image: np.array, mask: np.array):
        """
        Drawing each single frame.
        """
        height, width, n_band = flow.shape
        if self.post_factor is None:
            self.post_factor = self.calib * 1000

        u, v = flow[:, :, 0], flow[:, :, 1]
        w = flow[:, :, 2] if self.use_stereo else None  # Stereo plotting option

        # Image masking
        if self.color_type in ["vort", "shear", "normal"]:
            vort, shear, normal = utils.calc_vorticity(flow, calib=self.post_factor)
            if self.color_type == "vort":
                flo_diff = vort
            elif self.color_type == "shear":
                flo_diff = shear
            else:
                flo_diff = normal

            flo_diff = flo_diff / np.max(flo_diff) if self.maxmotion is None else flo_diff  # No maxflow input
            flo_rgb = np.uint8(self.scalar_map.to_rgba(flo_diff) * 255)[:, :, :3]
            flo_alpha = np.abs(flo_diff)  # Setting up the transparency mask
        elif self.color_type == "mag":
            flo_alpha = np.linalg.norm(flow, axis=-1)  # Calculate the flow magnitude
            flo_alpha = flo_alpha / np.max(flo_alpha) if self.maxmotion is None else flo_alpha  # No maxflow input
            flo_rgb = np.uint8(self.scalar_map.to_rgba(flo_alpha) * 255)[:, :, :3]
        else:
            flo_alpha = np.linalg.norm(flow, axis=-1)  # Calculate the flow magnitude
            flo_rgb = colorflow.motion_to_color(flow, maxmotion=self.maxmotion)

        flo_alpha[~mask] = 0

        # Superpose the image and flow color visualization if use_color is activated
        if self.use_color == 0:
            image[mask] = 255
            self.im1.set_data(image)
        elif self.use_color == 1:
            flo_rgb[~mask] = 0  # Merging image and plot the result
            image[mask] = 0
            image += flo_rgb
            self.im1.set_data(image)
        elif self.use_color == 2:
            flo_rgb[~mask] = 255
            alpha = np.uint8((flo_alpha - np.min(flo_alpha)) * 255 / (np.max(flo_alpha) - np.min(flo_alpha)))
            # alpha = np.uint8(flo_alpha * 255 / self.maxmotion)
            flo_rgba = np.dstack([flo_rgb, alpha])

            self.im1.set_data(image)
            self.im2.set_data(flo_rgba)
        else:
            self.im1.set_data(image)

        # TODO: Check the image plotting for stereo viz
        # self.im_stereo.set_facecolors(image.reshape(-1, 3) / 255) if self.use_stereo else None

        # Adding quiver plot (if necessary)
        self._q_plot(mask, u, v, w) if self.use_quiver else None

    def _q_plot(self, mask, u, v, w: Optional[np.array] = None) -> None:
        """
        Quiver plot config.
        params:
            u: Flow displacement at x-direction.
            v: Flow displacement at y-direction.
            mask: The masking array.
        """
        # Normalizing the flow
        height, width = u.shape
        u, v = u / self.velocity_factor, v / self.velocity_factor

        # Vector slicing (flow and mask)
        up, vp = u[::self.vector_step, ::self.vector_step], v[::self.vector_step, ::self.vector_step]
        maskp = mask[::self.vector_step, ::self.vector_step]
        mag = np.hypot(up, vp)

        # Stereo flo visualization option
        if self.use_stereo:  #TODO: Test the quiver3d vector update for Stereo plotting!
            assert w is not None
            wp = w[::self.vector_step, ::self.vector_step] / self.velocity_factor
            mag_stereo = np.linalg.norm(np.dstack([up, vp, wp]), axis=-1)
            up[~maskp], vp[~maskp], wp[~maskp] = np.nan, np.nan, np.nan

            x, y, z = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5, 0)
            xp = x[::self.vector_step, ::self.vector_step]
            yp = y[::self.vector_step, ::self.vector_step]
            zp = z[::self.vector_step, ::self.vector_step]
            segs = (xp, yp, zp, np.expand_dims(up, axis=-1), np.expand_dims(vp, axis=-1), np.expand_dims(wp, axis=-1))
            segs_flatten = np.array(segs).reshape(6, -1)
            segs_new = [[[xs, ys, zs], [us, vs, ws]] for xs, ys, zs, us, vs, ws in zip(*segs_flatten.tolist())]
            self.quiver_stereo.set_segments(segs_new)

            if not self.use_color:  #TODO: Test the quiver3d COLOR update!
                mag_rgb = np.uint8(self.scalar_map.to_rgba(mag_stereo))[:, :, :3]
                self.quiver_stereo.set_color(mag_rgb.reshape(-1, 3))
        else:
            up[~maskp], vp[~maskp] = np.nan, np.nan
            mag_stereo = mag

        # Updating quiver plot with the latest vector values
        if not self.use_color:
            self.quiver.set_UVC(up, -vp, C=self.norm(mag_stereo))
        else:
            self.quiver.set_UVC(up, -vp)


def get_image(basename, imdir, crop_window: Union[int, Tuple[int, int, int, int]] = 0) -> np.array:
    imname = basename + ".tif"
    impath = os.path.join(imdir, imname)
    img = imageio.imread(impath)
    if len(img.shape) < 3:
        img = np.dstack([img, img, img])

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


def color_map(resolution: int = 256, maxmotion: float = 1.0, show: bool = False, filename: Optional[str] = None,
              velocity_factor: float = 1.0) -> None:
    """
    Plotting the color code
    params:
        resolution: Colormap image resolution.
        maxmotion: Maximum flow motion.
        show: Option to display the plot or not.
        filename: Full file path to save the plot; use None for not saving the plot!
        resolution: To calibrate flow velocity from [pixel/frame] into [mm/second]
    Returns the flow colormap in mm/second.
    """
    # Init.
    bname = os.path.basename(filename).rsplit("_", 1)[0]

    # Color plot
    pts = np.linspace(-maxmotion, maxmotion, num=resolution)
    x, y = np.meshgrid(pts, pts)
    flo = np.dstack([x, y]) * velocity_factor  # Calibrating into real flow vector
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


def filter_plot(csvname: str, key: List[str], avg_window: int = 1, layered: bool = False, show: bool = False,
                filename: Optional[str] = None, title: Optional[str] = None, xlim: Optional[Tuple[float]] = None,
                ) -> Tuple[pd.DataFrame, plt.Axes]:
    """
    Applying moving average for signal filtering, and plot it
    """
    df = pd.read_csv(csvname, index_col=0)
    df_roll = df[key].rolling(window=avg_window, min_periods=1).mean()

    df_plot = df[['time']]
    col_plot = []

    for k in key:
        df_plot[k] = df[k]
        new_key = f"filtered_{k}" if layered else k
        df_plot[new_key] = df_roll[k]

        col_plot.append(k)
        col_plot.append(new_key) if new_key not in col_plot else None

    if layered and filename:
        fname, fext = os.path.splitext(filename)
        filename = fname + f"-layered{fext}"

    ax = df_plot.plot(x='time', y=col_plot)
    ax.set_title(title) if title is not None else None
    ax.set_xlabel("Time stamp [s]")
    ax.set_ylabel("Flow magnitude [mm/s]")
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0, xmax=df_plot['time'].max())

    plt.show() if show else None
    plt.savefig(filename, dpi=300, bbox_inches='tight') if filename else None

    return df_plot, ax


if __name__ == "__main__":
    # Plot maximum flow
    max_path = "./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif/report/Test 03 L3 NAOCL 22000 fpstif_max.csv"
    maxplot_path = "./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif/report/Test 03 L3 NAOCL 22000 fpstif_max.png"
    filter_plot(max_path, avg_window=80, key=["maxflo"], filename=maxplot_path, layered=False,
                title="Change of maximum flow magnitude")

    # Generate colormap
    color_map(resolution=256, maxmotion=4, show=True)

    print("DONE!")
