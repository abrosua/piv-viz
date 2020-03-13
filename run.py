import os
import sys
from glob import glob
from typing import Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

pivdir = "/home/faber/thesis/thesis_faber"
sys.path.append(pivdir)
from inference import Inference, flowname_modifier, write_flow, piv_liteflownet, hui_liteflownet
from src.utils_plot import quiver_plot


def getpair(dir: str, n_images: int = 2, extensions: Tuple[str] = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'ppm')):

	img_files = []
	for ext in extensions:
		img_files += sorted(glob(os.path.join(dir, f'*.{ext}')))

	return img_files[:n_images]


if __name__ == '__main__':
	dir = "./frames/Test 03 L3 NAOCL 22000 fpstif"
	loopdir = [None, 5, 10, 100]

	# Net init,
	num_images = 2
	write = True
	savefig = True
	modeldir = "models/torch/Hui-LiteFlowNet.paramOnly"
	args_model = os.path.join(pivdir, modeldir)

	if os.path.isfile(args_model):
		weights = torch.load(args_model)
	else:
		raise ValueError('Unknown params input!')

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	net = piv_liteflownet(weights).to(device)

	for i in loopdir:
		inputdir = dir + f"_{i}" if i else dir
		imnames = getpair(inputdir, n_images=num_images)

		outdir = os.path.join("results", os.path.basename(inputdir))
		os.makedirs(outdir) if not os.path.isdir(outdir) else None

		prev_frame = None
		for curr_frame in tqdm(imnames, ncols=100, leave=True, unit='pair', desc=f'Evaluating {inputdir}'):
			if prev_frame is not None:
				out_flow = Inference.parser(net,
											Image.open(prev_frame).convert('RGB'),
											Image.open(curr_frame).convert('RGB'),
											device=device)
				# Post-processing here
				out_name = flowname_modifier(prev_frame, outdir, pair=False)
				if write:
					write_flow(out_flow, out_name)

				# Save quiver plot
				if savefig:
					figname = os.path.splitext(out_name)[0] + ".png"
					quiver_plot(out_flow, filename=figname)

			prev_frame = curr_frame

		tqdm.write(f'Finish processing all images from {inputdir} path!')
