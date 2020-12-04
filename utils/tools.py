import os
import shutil
from glob import glob
from typing import Tuple, Optional, Union

import cv2


def split_abspath(path: str, cutting_path: Optional[Union[str, int]] = None) -> Tuple[str, str]:
	cleanpath = os.path.normpath(os.path.abspath(path))
	cleanpath_sep = cleanpath.split(os.sep)

	if cleanpath_sep[0] == "":  # To join the List as an Absolute path
		cleanpath_sep[0] = os.sep

	if cutting_path is None:
		frontpath_sep = cleanpath_sep
		backpath_sep = [""]
	elif type(cutting_path) is int:
		frontpath_sep = cleanpath_sep[:cutting_path]
		backpath_sep = cleanpath_sep[cutting_path:]
	elif type(cutting_path) is str:
		frontpath_sep = cleanpath_sep[:cleanpath_sep.index(cutting_path)]
		backpath_sep = cleanpath_sep[cleanpath_sep.index(cutting_path):]
	else:
		raise ValueError(f"UNKNOWN format of the 'cutting_path' ({cutting_path}) input!")

	return os.path.join(*frontpath_sep), os.path.join(*backpath_sep)


def getpair(dir: str, n_images: int = 2, start_at: int = 0,
			extensions: Tuple[str] = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'ppm')):
	assert n_images != 1

	img_files = []
	for ext in extensions:
		img_files += sorted(glob(os.path.join(dir, f'*.{ext}')))

	if n_images < 0:
		return img_files[start_at:]
	else:
		return img_files[start_at : start_at+n_images]


def file_naming(input_name: str, subfolder=None):
	basename, _ = os.path.splitext(os.path.basename(input_name))
	if subfolder is None:
		dirname = os.path.join('frames', basename)
	else:
		dirname = os.path.join('frames', subfolder, basename)

	print(f"The results will be stored in {dirname}")
	if not os.path.isdir(dirname):
		os.makedirs(dirname)

	return dirname, basename


def copyfile(dir: str, targetdir: str, skip_count: int = 1,
			 extensions: Tuple[str] = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'ppm')):
	if not os.path.isdir(targetdir):
		os.makedirs(targetdir)

	img_files = []
	for ext in extensions:
		img_files += sorted(glob(os.path.join(dir, f'*.{ext}')))

	for img in sorted(img_files):
		imname, _ = os.path.splitext(os.path.basename(img))
		imindex = int(imname.rsplit('_', 1)[-1])

		if imindex % skip_count == 0:
			shutil.copy2(img, targetdir)


class Sketcher:
	def __init__(self, windowname, dests, colors_func):
		self.prev_pt = None
		self.windowname = windowname
		self.dests = dests
		self.colors_func = colors_func
		self.dirty = False
		self.show()
		cv2.setMouseCallback(self.windowname, self.on_mouse)

	def show(self):
		cv2.imshow(self.windowname, self.dests[0])

	def on_mouse(self, event, x, y, flags, param):
		pt = (x, y)
		if event == cv2.EVENT_LBUTTONDOWN:
			self.prev_pt = pt
		elif event == cv2.EVENT_LBUTTONUP:
			self.prev_pt = None

		if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
			for dst, color in zip(self.dests, self.colors_func()):
				cv2.line(dst, self.prev_pt, pt, color, 5)
			self.dirty = True
			self.prev_pt = pt
			self.show()


if __name__ == '__main__':
	skip = 5
	inputdir = "./frames/Test 03 L3 NAOCL 22000 fpstif"
	copydir = f"./frames/Test 03 L3 NAOCL 22000 fpstif_{skip}"

	copyfile(inputdir, copydir, skip_count=skip)
