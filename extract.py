import cv2
import os
import shutil

from tqdm import tqdm
from PIL import Image, ImageSequence

from utils import tools


def from_multitiff(multipage_tiff, skip_count=1, start_at=0):
	dirname, basename = tools.file_naming(multipage_tiff)
	multi_img = Image.open(multipage_tiff)

	# Iterate images from the multi-tiff image
	idx = start_at
	print(f"Start iterating from image #{idx}")

	for i, page in tqdm(enumerate(ImageSequence.Iterator(multi_img)), desc=basename, unit='frames'):
		idx = i + start_at
		filename = f"{basename}_{str(idx).zfill(5)}.tif"

		if i % skip_count == 0:
			page.save(os.path.join(dirname, filename))
	print(f"Finished iterating at image #{idx}")

def from_gif(gifpath, skip_count=1, keep_origin=True, ext: str = "tif"):
	dirname, basename = tools.file_naming(gifpath)
	frame = Image.open(gifpath)
	nframes = 0

	while frame:
		framename = f"{basename}_{str(nframes).zfill(5)}.{ext}"
		if nframes % skip_count == 0:
			frameRgb = frame if keep_origin else frame.convert("RGB")
			frameRgb.save(os.path.join(dirname, framename))
		nframes += 1

		try:
			frame.seek(nframes)
		except EOFError:
			break

	print(f"Finished extracting frame(s) from '{gifpath}' at '{dirname}'!")


def copyfile(sourcedir, targetdir):
	if not os.path.isdir(targetdir):
		os.makedirs(targetdir)

	if os.path.isdir(sourcedir):
		print(f"Copying file(s) from '{sourcedir}' to '{targetdir}'")
	else:
		return print(f"The source directory ('{sourcedir}') is NOT found! Skipping the process...")

	sourcefiles = sorted([f for f in os.listdir(sourcedir) if os.path.isfile(os.path.join(sourcedir, f))])
	tqdm.write(f"Founded {len(sourcefiles)} images in '{sourcedir}'")
	i = 0

	for i, src in tqdm(enumerate(sourcefiles), desc=sourcedir, unit="frames"):
		srcfile = os.path.join(sourcedir, src)
		id = src.rsplit("_", 1)[1]

		fname = os.path.basename(targetdir) + "_" + id
		targetname = os.path.join(targetdir, fname)

		# Copying files
		shutil.copy2(srcfile, targetname)

	tqdm.write(f"Finished copying {i+1} file(s)!")


if __name__ == "__main__":
	# display_frames("./endo-raw/Test 03 L3 NAOCL 22000 fps.cine", count=100, skip_count=1)

	multitiff_file = "./endo-raw/multitif/Test 06 EDTA EA Full 22000 fps/Test 06 EDTA EA Full 22000 fps-05.tif"
	# from_multitiff(multitiff_file)

	gifpath = "./frames/meme.gif"
	# gifpath = "./results/present/Test 03 L3 NAOCL 22000 fpstif-8600_90_tqvid.gif"
	# from_gif(gifpath, keep_origin=False, ext="png")

	# Copying file
	targetdir = "./frames/Test 06 EDTA EA Full 22000 fps"
	for i in reversed(range(6)):
		sourcedir = f"./frames/Test 06 EDTA EA Full 22000 fps-0{i}"
		copyfile(sourcedir, targetdir)
