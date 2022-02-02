source = "archive/img_align_celeba"
downloaded = "formatted/downscaled"
target = "formatted/target"

import os
import face_recognition
from PIL import Image
import numpy as np

imgs = os.listdir(source)  # Reading in images from the file system


# Idea behind downsampling
# You take a block of pixels (let's say 4 x 4) and you just average the pixel values
# Use the averaged pixel values to construct the downsampled image

def downscale(img, sf):
	w, h = img.shape[:2]  # Gets width and height of image
	nw, nh = w // sf, h // sf  # sf is the scaling factor (how much to scale by)

	dtype = img.dtype.type
	nimg = np.zeros((nw, nh, img.shape[2]), dtype=dtype)  # img.shape[2] is the img channels (like R, G, B)

	# Averaging blocks of pixels into nimg

	return nimg


for filename in imgs:
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
