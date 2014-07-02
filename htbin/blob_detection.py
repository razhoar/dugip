from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
import skimage
import skimage.draw

import numpy as np

import skimage.io as IO
import os
import os.path


def load_image_from_file(filename):
   filename = os.path.abspath(filename)
   
   image = data.load(filename)
   image_gray = rgb2gray(image)

   return image, image_gray


def detectar_blobs(image):
   blobs = blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)
   # Compute radii in the 3rd column.
   blobs[:, 2] = blobs[:, 2] * sqrt(2)

   return blobs

def pintar_cuadrados(blobs, image_dst):
   for y, x, r in blobs:
      poly = np.array((
         (y+r, x+r),
         (y+r, x-r),
         (y-r, x-r),
         (y-r, x+r),
         (y+r, x+r)
         ))

      rr, cc = skimage.draw.polygon(poly[:, 0], poly[:, 1], image_dst.shape)
      image_dst[rr, cc, 1] = 255


if __name__ == '__main__':
	import sys, os
	input_dir, output_dir = sys.argv[1:]
	
	print "BLOB DETECTION**********************"
	print input_dir, output_dir
	
	for filename in os.listdir(input_dir):
		if os.path.isfile(filename):
			image, image_gray = load_image_from_file(filename)
			blobs = detectar_blobs(image_gray)
			pintar_cuadrados(blobs, image)
			
			dst_filename = os.path.join(output_dir, filename)
			IO.imsave(dst_filename, image)

