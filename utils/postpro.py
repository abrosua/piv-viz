import os
import json
from glob import glob
from typing import Optional, List, Tuple
import pathlib as Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from labelme.utils import shape_to_mask

import utils
import matplotlib
matplotlib.use('TkAgg')


class Label:
    def __init__(self, labelpath: str, flodir: str, verbose: int = 0) -> None:
        """
        Acquiring all the available label as a dictionary.
        params:
            labelpath:
            labeldir:
            verbose:
        """
        self.label_path = labelpath
        self.verbose = verbose
        self.label = {}

        # Obtaining the working directory
        flodir = utils.split_abspath(flodir)[0]
        workdir, _ = utils.split_abspath(flodir, cutting_path="results")

        # Importing the Labels
        if not os.path.isfile(labelpath):  # Label file checking
            raise ValueError(f"Label file at '{labelpath}' is NOT FOUND!")

        with open(labelpath) as json_file:
            data = json.load(json_file)
            self.img_shape = [data['imageHeight'], data['imageWidth']]

            # Image path
            _, endpath = utils.split_abspath(data["imagePath"], cutting_path="frames")
            self.img_dir = endpath.split(os.sep)[1]  # Take the path after "frames"
            self.img_path = os.path.join(workdir, endpath)
            imname = os.path.basename(self.img_path)

            for d in data['shapes']:
                labelkey = d['label']

                if labelkey not in self.label.keys():
                    self.label[labelkey] = {'points': [d['points']],
                                            'shape_type': d['shape_type']}
                else:
                    self.label[labelkey]['points'].append(d['points'])

        # Importing the Flow files
        flocheck = os.path.basename(flodir).lower()
        if flocheck == "flow":
            bname = os.path.splitext(imname)[0]
            ext = "_out.flo"
        else:
            bname = imname.rsplit("-", 1)[0]
            ext = f"-{flocheck[0].upper()}_out.flo"

        floname = bname + ext
        self.flopath = os.path.join(flodir, floname)

        if not os.path.isfile(self.flopath):  # Flow file checking
            raise ValueError(f"Flow file at '{self.flopath}' is NOT FOUND!")

    def get_column(self) -> Optional[float]:
        """
        Returning the lowest point of the air column occurrence.
        """
        if 'column' in self.label.keys():
            points = np.array(self.label['column']['points'][0])
            y_points = points[:, 1]
            return -y_points.min()
        else:
            print(f"The Air Column label is NOT found in '{self.label_path}'") if self.verbose else None
            return None

    def get_flo(self, key, fill_with: Optional[float] = None) -> Tuple[Optional[np.array], Optional[np.array]]:
        """
        Acquiring the masked flow vector and its respective mask array.
        params:
            key: The label key of the flow to obtain (e.g., 'flow', 'v1', 'v2')
            fill_with: Filling value to the masked vector.
        """
        if key in self.label.keys():
            flow_label = self.label[key]

            # Flow init.
            out_flow = utils.read_flow(self.flopath)
            mask = np.full(out_flow.shape[:2], False)
            mask_flow = np.full(out_flow.shape, fill_with) if fill_with is not None else out_flow

            # Filling the masked flow array
            for flow_point in flow_label['points']:
                mask += shape_to_mask(self.img_shape, flow_point, shape_type=flow_label['shape_type'])

            mask_flow[mask] = out_flow[mask]
            return mask_flow, mask
        else:
            print(f"The '{key}' label is NOT found in '{self.label_path}'") if self.verbose else None
            return None, None


def velo_mean(flo: np.array, mask: Optional[np.array] = None):
    if mask is None:
        flo_mag = np.linalg.norm(flo, axis=-1)
        flo_mag_clean = flo_mag[~np.isnan(flo_mag)]
    else:
        flo_clean = flo[mask]
        flo_mag_clean = np.linalg.norm(flo_clean, axis=-1)

    return np.mean(flo_mag_clean)


def checkstat(data):
    sns.distplot(data, hist=True, kde=True,
                 bins=int(180 / 5), color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})
    plt.show()


def region_velo(labelpath: str, netname: str, flodir: str, key: str, fps: int = 1, start_at: int = 0, end_at: int = -1,
                num_flows: int = -1, avg_step: int = 1, show: bool = False, filename: Optional[str] = None,
                calibration_factor: float = 1.0, verbose: int = 0) -> np.array:
    """
    Get regional velocity data (either v1 or v2).
    params:
        labelpath: The base label file to use.
        netname: Network name result to use.
        flodir: Main directory of the flow.
        key: Which regional flow to choose (v1 or v2).
        fps: Image frame frequency (Frame Per Second).
        start_at: Flow index to start.
        end_at: Last flow index (if -1, use the last flow in the flowdir).
        num_flows: Number of flows to choose (if -1, use all the available flows in the flowdir).
        avg_step: Number of steps to average the flow value (if 1, calculate instantaneous velocity instead).
        calibration_factor: To calibrate pixel to (estimated) real displacement.
        verbose: The verbosal option
    returns:
        numpy array of the regional velocity summary at each time frame, in terms of average 2d velocity and magnitude.
        The flow regional velocity is in mm/second.
    """
    # Init.
    assert os.path.isfile(labelpath)
    assert os.path.isdir(flodir)
    assert avg_step > 0

    # Flow metadata
    flopaths = sorted(glob(os.path.join(flodir, "*.flo")))
    nflow = len(flopaths)
    end_at = nflow if end_at < 0 else end_at
    num_flows = nflow - start_at if num_flows < 0 else num_flows

    step = int(np.floor(end_at / num_flows))
    idx = list(range(start_at, end_at, step))
    key_title = "instantaneous" if avg_step == 1 else "mean"

    # Getting the v1/v2 label
    label = Label(labelpath, netname, verbose=verbose)
    flow_label = label.label[key]

    # Iterate over the flows
    velo_record = [[0.0, [0.0, 0.0], 0.0]]

    for id in tqdm(idx, desc=f"Flow at {key}", unit="frame"):
        out_flow, _ = utils.read_flow_collection(flodir, start_at=id, num_images=avg_step)
        out_flow = out_flow * fps * calibration_factor  # Calibrating into mm/second

        out_mag = np.linalg.norm(out_flow, axis=-1)
        avg_flow, avg_mag = np.mean(out_flow, axis=0), np.mean(out_mag, axis=0)
        mask = np.full(avg_flow.shape[:2], False)

        # Filling the masked flow array
        for flow_point in flow_label['points']:
            mask += shape_to_mask(avg_flow.shape[:2], flow_point, shape_type=flow_label['shape_type'])

        velo_record.append([(id+1)/fps, np.mean(avg_flow[mask], axis=0), np.mean(avg_mag[mask], axis=0)])
    velo_record = np.array(velo_record)

    plt.plot(velo_record[:, 0], velo_record[:, -1])
    plt.title(f"{key} {key_title} velocity at each time frame")
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlabel("Time stamp [frame]")
    plt.ylabel(f"{key} velocity [pix]")

    plt.show() if show else None
    plt.savefig(filename, dpi=300, bbox_inches='tight') if filename else None
    plt.clf()

    return velo_record


def column_level(labelpaths: List[str], netname: str, fps: int = 1, calib: float = 1., show: bool = False,
                 filename: Optional[str] = None, verbose: int = 0, xlim: Optional[List[float]] = None,
                 add_title: bool = False,
                 ) -> Tuple[np.array, List[str], float, plt.Axes]:
    """
    Gathering a series of air column coordinates, to plot the change in air column level.
    params:
        labelpaths: List of path for Label files.
        netname: Name of the network.
        fps: The video frame frequency (frame/second)
        verbose: Verbosal option value.
    Returns an array of the change in air column level.
    """
    column, img_paths = [], []
    flowdir = ""

    for labelpath in labelpaths:
        flowdir = os.path.basename(os.path.dirname(labelpath))
        idx = int(str(os.path.splitext(labelpath)[0].rsplit("_", 1)[1]))
        time_frame = (idx+1)/fps

        label = Label(labelpath, netname, verbose=verbose)
        column_tmp = label.get_column()

        if column_tmp is None:
            continue
        column.append([time_frame, column_tmp * calib])
        img_paths.append(label.img_path)

    column_mat = np.array(column)
    init_point = column_mat[0, 1]
    column_mat[:, 1] -= init_point  # Each level is relative to the initial condition!

    # Define plotting
    fig, ax = plt.subplots()
    ax.plot(column_mat[:, 0], column_mat[:, 1])
    ax.set_title(f"Column level change of {flowdir}") if add_title else None
    ax.set_ylim(top=0)
    ax.set_xlim(left=0) if xlim is None else plt.xlim(xlim)
    ax.set_xlabel("Time stamp [s]")
    ax.set_ylabel("Relative column level [mm]")

    plt.show() if show else None
    plt.savefig(filename, dpi=300, bbox_inches='tight') if filename else None

    return column_mat, img_paths, -init_point, ax


def get_max_flow(flodir: str, labelpath: Optional[str] = None, start_at: int = 0, end_at: int = -1,
                 filename: Optional[str] = None, aggregate: Tuple[str, ...] = ('max'), calib: float = 1.0, fps: int = 1,
                 verbose: int = 0) -> Tuple[float, np.array]:
    """
    Get maximum flow magnitude within the flow direction.
    params:
        flodir: Flow directory.
        labelpath: Label file input to mask the flow, optional.
        start_at: Starting index.
        end_at: Ending index.
    Returns the maximum flow magnitude.
    """
    # Init.
    assert os.path.isdir(flodir)
    name_list = utils.split_abspath(flodir).split(os.sep)
    floname, netname = str(name_list[-2]), str(name_list[-3])
    velo_factor = fps * calib

    if filename:
        savedir = os.path.dirname(filename)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)

    if labelpath is not None:
        assert os.path.isfile(labelpath)
        mask_label = Label(labelpath, netname, verbose=verbose).label["video"]
    else:
        mask_label = None

    flopaths_raw = sorted(glob(os.path.join(flodir, "*.flo")))
    end_at = len(flopaths_raw) if end_at < 0 else end_at
    flopaths = flopaths_raw[start_at:end_at]

    # Iterate over the flopaths
    max_flo, vort_flo, shear_flo, normal_flo = 0.0, 0.0, 0.0, 0.0
    data_flos = {"flow": [[0.0] * (len(aggregate) + 1)],
                 "vort": [[0.0] * (len(aggregate)*2 + 1)],
                 "shear": [[0.0] * (len(aggregate)*2 + 1)],
                 "normal": [[0.0] * (len(aggregate)*2 + 1)],
                 }

    for i, flopath in enumerate(tqdm(flopaths, desc=f"Max flow at {floname}", unit="frame")):
        flow = utils.read_flow(flopath)
        flow = flow * velo_factor  # Calibrate flow into real velocity
        vort, shear, normal = utils.calc_vorticity(flow, calib=calib)
        time_id = (i + 1) / fps

        if mask_label is not None:
            mask = shape_to_mask(flow.shape[:2], mask_label["points"][0], shape_type=mask_label['shape_type'])
            flow, vort, shear, normal = flow[mask], vort[mask], shear[mask], normal[mask]

        mag_flo = np.linalg.norm(flow, axis=-1)
        max_flo = np.max(mag_flo) if np.max(mag_flo) > max_flo else max_flo
        vort_flo = np.max(np.abs(vort)) if np.max(np.abs(vort)) > vort_flo else vort_flo
        shear_flo = np.max(np.abs(shear)) if np.max(np.abs(shear)) > shear_flo else shear_flo
        normal_flo = np.max(np.abs(normal)) if np.max(np.abs(normal)) > normal_flo else normal_flo

        agg_flo, agg_vor, agg_shear, agg_normal = [time_id], [time_id], [time_id], [time_id]
        for agg in aggregate:
            agg_module = getattr(np, agg)
            agg_flo.append(agg_module(mag_flo))
            agg_vor.extend([agg_module(vort[vort>0]), agg_module(np.abs(vort[vort<0]))])
            agg_shear.extend([agg_module(shear[shear > 0]), agg_module(np.abs(shear[shear < 0]))])
            agg_normal.extend([agg_module(normal[normal > 0]), agg_module(np.abs(normal[normal < 0]))])

        data_flos["flow"].append(agg_flo)
        data_flos["vort"].append(agg_vor)
        data_flos["shear"].append(agg_shear)
        data_flos["normal"].append(agg_normal)

    col_name, col_name_de = ['time'], ['time']
    for k in aggregate:
        col_name.append(k)
        col_name_de.extend([f"{k}_p", f"{k}_n"])

    data_flos_df = {}
    for k, v in data_flos.items():
        data_flos_tmp = pd.DataFrame(np.array(v), columns=col_name) if k == "flow" else \
            pd.DataFrame(np.array(v), columns=col_name_de)
        data_flos_df[k] = data_flos_tmp

        if filename:
            filename_tmp, ext = os.path.splitext(filename)
            filepath = filename_tmp + f"-{k}{ext}"
            data_flos_tmp.to_csv(filepath, index_label="frame")

    if verbose:
        tqdm.write(f"Maximum flow at {floname} (from frame {start_at} to {end_at}) is {max_flo:.2f}")
        tqdm.write(f"Maximum vorticity at {floname} (from frame {start_at} to {end_at}) is {vort_flo:.2f}")
        tqdm.write(f"Maximum shear stress at {floname} (from frame {start_at} to {end_at}) is {shear_flo:.2f}")
        tqdm.write(f"Maximum normal stress at {floname} (from frame {start_at} to {end_at}) is {normal_flo:.2f}")

    return max_flo, data_flos_df


if __name__ == '__main__':
    flodir = "./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif/flow"
    imdir = "./frames/Test 03 L3 NAOCL 22000 fpstif"

    # <------------ Use flowviz (uncomment for usage) ------------>
    # use_flowviz(flodir, imdir, start_at=4900, num_images=50, lossless=False)

    # <------------ Use get_max_flow (uncomment for usage) ------------>
    labelpath = "./labels/Test 03 L3 NAOCL 22000 fpstif/Test 03 L3 NAOCL 22000 fpstif_13124.json"
    max_path = "./results/Hui-LiteFlowNet/Test 03 L3 NAOCL 22000 fpstif/report/Test 03 L3 NAOCL 22000 fpstif_max.csv"
    max_flow, max_flows = get_max_flow(flodir, labelpath, verbose=1, filename=max_path, aggregate=("max", "mean"))

    # <------------ Use get_label (uncomment for usage) ------------>
    netname = "Hui-LiteFlowNet"
    labelpaths = sorted(glob(os.path.join("./labels/Test 03 L3 NAOCL 22000 fpstif", "*")))

    # Variable init.
    v1, v2, flow, air_column = {}, {}, {}, {}

    for labelpath in labelpaths:
        labels = Label(labelpath, netname)

        v1_tmp, _ = labels.get_flo('v1', fill_with=np.nan)
        if v1_tmp is not None:
            v1[labelpath] = v1_tmp

        v2_tmp, _ = labels.get_flo('v2', fill_with=np.nan)
        if v2_tmp is not None:
            v2[labelpath] = v2_tmp

        flow_tmp, _ = labels.get_flo('flow', fill_with=np.nan)
        if flow_tmp is not None:
            flow[labelpath] = flow_tmp

        air_column_tmp = labels.get_column()
        if air_column_tmp is not None:
            air_column[labelpath] = air_column_tmp

    # Data post-processing (i.e., calculate mean, deviation, etc)
    # v1_mean = velo_mean(v1['flo'], mask=v1['mask'])

    print('DONE!')
