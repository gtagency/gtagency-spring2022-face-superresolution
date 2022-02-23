source = "archive/img_align_celeba"
downscaled = "formatted/downscaled"
target = "formatted/target"

import os
import face_recognition
from PIL import Image
import numpy as np

imgs = os.listdir(source)  # Reading in images from the file system


def downscale(img, sf):
	"""
	Idea behind downsampling
	You take a block of pixels (let's say 4 x 4) and you just average the pixel values
	Use the averaged pixel values to construct the downsampled image

	:param img:	Image in array format
	:param sf: Scaling factor
	:return: Downsampled image in resized format
	"""
	w, h = img.shape[:2]  # Gets width and height of image
	nw, nh = w // sf, h // sf  # Scale image dimensions down by sf

	dtype = img.dtype.type

	# Create numpy array of zeros with dimension nw x nh x 3
	nimg = np.zeros((nw, nh, img.shape[2]), dtype=img.dtype.type)  # img.shape[2] is the img channels (like R, G, B)

	# Averaging blocks of pixels into nimg
	for i in range(nw):
		for j in range(nh):
			for c in range(img.shape[2]):
				nimg[i, j, c] = dtype(
					img[sf * i:sf * (i + 1), sf * j:sf * (j + 1), c].mean())  # Neil +0.5 for rounding?

	return nimg


for index, filename in enumerate(imgs):
	# Crop images to 128 x 128 around the face
	im = face_recognition.load_image_file(os.path.join(source, filename))
	faces = face_recognition.face_locations(im)
	if len(faces) == 0:
		continue

	face = faces[0]
	top, right, bot, left = face  # Corners of the face we want to crop around

	# Find midway point
	v = (top + bot) // 2
	h = (right + left) // 2

	crop = im[v - 64: v + 64, h - 64: h + 64]  # Image splicing, go 64 in each direction around the center of face

	# Sanity check
	if crop.shape != (128, 128, 3):
		continue

	x32 = downscale(crop, 4)
	x16 = downscale(x32, 2)

	# Create image from array
	x32 = Image.fromarray(x32)
	x16 = Image.fromarray(x16)

	x16.save(os.path.join(downscaled, "%03d.png" % index))
	x32.save(os.path.join(target, "%03d.png" % index))

	# Don't want to use entire dataset
	if index >= 10:
		break
