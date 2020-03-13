import cv2
import os

from PIL import Image, ImageSequence
from pycine.color import color_pipeline, resize
from pycine.raw import read_frames
from tqdm import tqdm

from utils import file_naming


def display_frames(cine_file, start_frame=1, count=1, skip_count=1):
	basename, dirname = file_naming(cine_file)

	raw_images, setup, bpp = read_frames(cine_file, start_frame=start_frame, count=count)
	# rgb_images = (color_pipeline(raw_image, setup=setup, bpp=bpp) for raw_image in raw_images)

	for i, rgb_image in enumerate(raw_images):
		frame = start_frame + i

		if i % skip_count == 0:
			filename = f"{basename}_{str(frame).zfill(4)}.tif"

			if setup.EnableCrop:
				rgb_image = rgb_image[
							setup.CropRect.top:setup.CropRect.bottom + 1,
							setup.CropRect.left:setup.CropRect.right + 1
							]

			if setup.EnableResample:
				rgb_image = cv2.resize(rgb_image, (setup.ResampleWidth, setup.ResampleHeight))

			# Saving image
			savename = os.path.join(dirname, filename)
			save_img = rgb_image * 255
			cv2.imwrite(savename, save_img)

			jpeg_img = resize(save_img, 720)
			# cv2.imwrite(os.path.join(dirname, 'resize.tif'), jpeg_img)
			# cv2.imencode('.jpg', jpeg_img)


def extract_multitiff(multipage_tiff, skip_count=1):
	basename, dirname = file_naming(multipage_tiff)
	multi_img = Image.open(multipage_tiff)

	for i, page in tqdm(enumerate(ImageSequence.Iterator(multi_img)), desc=basename, unit='frames'):
		filename = f"{basename}_{str(i).zfill(4)}.tif"

		if i % skip_count == 0:
			page.save(os.path.join(dirname, filename))


if __name__ == "__main__":
	# display_frames("./endo-raw/Test 03 L3 NAOCL 22000 fps.cine", count=100, skip_count=1)

	multitiff_file = "./endo-raw/tes/Test 03 L3 NAOCL 22000 fpstif.tif"
	extract_multitiff(multitiff_file)
