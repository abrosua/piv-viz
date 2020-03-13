import sys
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

pivdir = "/home/faber/thesis/thesis_faber"
sys.path.append(pivdir)
from src.utils_plot import read_flow, quiver_plot, motion_to_color


def singleplot(floname, imdir):
	# Generate image name from the flo file
	basename = str(os.path.basename(floname).rsplit('_', 1)[0])
	subdir = os.path.basename(os.path.dirname(floname))
	imname = os.path.join(imdir, subdir, basename + ".png")

	plt.imshow(imname)
	out_flow = read_flow(floname)
	quiver_plot(out_flow, show=True)


if __name__ == '__main__':
	imdir = "./frames"
	floname = "./results/Test 03 L3 NAOCL 22000 fpstif_100/Test 03 L3 NAOCL 22000 fpstif_0000_out.flo"
	singleplot(floname, imdir)
