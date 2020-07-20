import cv2
import os

from tqdm import tqdm
from PIL import Image, ImageSequence

from utils import tools


def from_multitiff(multipage_tiff, skip_count=1):
	dirname, basename = tools.file_naming(multipage_tiff)
	multi_img = Image.open(multipage_tiff)

	# Iterate images from the multi-tiff image
	for i, page in tqdm(enumerate(ImageSequence.Iterator(multi_img)), desc=basename, unit='frames'):
		filename = f"{basename}_{str(i).zfill(5)}.tif"

		if i % skip_count == 0:
			page.save(os.path.join(dirname, filename))


def from_gif(gifpath, skip_count=1, keep_origin=True):
	dirname, basename = tools.file_naming(gifpath)
	frame = Image.open(gifpath)
	nframes = 0

	while frame:
		framename = f"{basename}_{str(nframes).zfill(5)}.tif"
		if nframes % skip_count == 0:
			frameRgb = frame if keep_origin else frame.convert("RGB")
			frameRgb.save(os.path.join(dirname, framename))
		nframes += 1

		try:
			frame.seek(nframes)
		except EOFError:
			break

	print(f"Finished extracting frame(s) from '{gifpath}' at '{dirname}'!")


if __name__ == "__main__":
	# display_frames("./endo-raw/Test 03 L3 NAOCL 22000 fps.cine", count=100, skip_count=1)

	multitiff_file = "./endo-raw/multitif/Test 02 L3 EDTA ND TIP 22000 fpstif.tif"
	# from_multitiff(multitiff_file)

	gifpath = "./frames/meme.gif"
	from_gif(gifpath, keep_origin=False)
