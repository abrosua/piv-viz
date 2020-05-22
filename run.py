import os
from glob import glob
from typing import Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import tools
from utils import Inference, hui_liteflownet, piv_liteflownet, flowname_modifier, write_flow

maindir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(maindir)
pivdir = os.path.join(os.path.dirname(maindir), "thesis_faber")


if __name__ == '__main__':
	# Run config
	write = True

	# Input frames init.
	start_id = 0
	num_images = -1
	dir = "./frames/Test 02 L3 EDTA END TIP 22000 fpstif"
	loopdir = [None]  # [None, 5, 10, 100]

	# Net init,
	modeldir = "models/pretrain_torch/Hui-LiteFlowNet.paramOnly"
	# modeldir = "models/pretrain_torch/PIV-LiteFlowNet-en.paramOnly"
	args_model = os.path.join(pivdir, modeldir)

	if os.path.isfile(args_model):
		weights = torch.load(args_model)
		netname = os.path.splitext(os.path.basename(args_model))[0]
	else:
		raise ValueError('Unknown params input!')

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	net = hui_liteflownet(weights).to(device)
	#net = piv_liteflownet(weights).to(device)

	out_names = []
	for i in loopdir:
		inputdir = dir + f"_{i}" if i else dir
		imnames = tools.getpair(inputdir, n_images=num_images, start_at=start_id)

		# Name checking
		num_images = "end" if num_images < 0 else num_images
		is_all_flow = (start_id == 0) and (num_images < 0)

		outsubdir = f"{os.path.basename(inputdir)}-{start_id}_{num_images}" if not is_all_flow \
			else os.path.basename(inputdir)
		outdir = os.path.join("./results", netname, outsubdir, "flow")
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
					out_names.append(out_name)

			prev_frame = curr_frame

		tqdm.write(f'Finish processing all images from {inputdir} path!')
