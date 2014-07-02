import numpy as np
import skimage.data
import skimage.transform
import sklearn.cluster

import numpy as np
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

import skimage.color
import skimage.io

from time import time

import argparse


def getplayerblobs (imgdata, scalefact = 0.5) :
   import numpy as np
   import skimage.data
   import skimage.transform
   import sklearn.cluster
   
   import numpy as np
   import pylab as pl
   from sklearn.cluster import KMeans
   from sklearn.metrics import euclidean_distances
   from sklearn.datasets import load_sample_image
   from sklearn.utils import shuffle
   
   import skimage.color
   import skimage.io
   
   from time import time
   
   imgdata = skimage.transform.rescale (imgdata, scalefact)
   
   w, h, d = original_shape = tuple(imgdata.shape)
   
   # Subsample (optional?)
   # Apply filters (optional?)
   # Convert image to hsv (optional?)
   # Get hsv histogram of color (optional?)
   image_array = imgdata.reshape((imgdata.shape[0]*imgdata.shape[1],3))
   ################################################################################# 
   # Option 1 :
   # With the histogram determinate the most prevalent color (the field) (HSV mode is needed here)
   # Make a thresholdt with the distance to that color to get the mask
   #################################################################################
   # Option 2 :
   # Cluster colors similar (k-means?)
   #pixels = pixels[np.random.randint(pixels.shape[0], size=10000), :]
   
   n_colors = 10
   image_array_sample = shuffle(image_array, random_state=0)[:1000]
   kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
   labels = kmeans.predict(image_array)
   
   def recreate_image(codebook, labels, w, h):
       """Recreate the (compressed) image from the code book & labels"""
       d = codebook.shape[-1]
       image = np.zeros((w, h, d))
       label_idx = 0
       for i in range(w):
           for j in range(h):
               #print labels[label_idx], label_idx
               image[i][j] = codebook[labels[label_idx]]
               label_idx += 1
       return image
   
   newimg = recreate_image(kmeans.cluster_centers_, labels, w, h)
   skimage.io.imsave ("/tmp/pepe.jpg",newimg)
   
   # Identify the cluster with more pixels (the field)
   labelsbyweigth = np.bincount(labels).argsort()[::-1]
   maxlabel = labelsbyweigth[0]
   labels[labels == labelsbyweigth[1]] = maxlabel
   maxlabels_list = labelsbyweigth[[0,1]]
   
   # Make simple mask
   maskcodebook = kmeans.cluster_centers_.copy()
   masklabels = labels.copy()
   
   mask = (masklabels==maxlabel).astype("int")
   #binary_codebook = np.array([[0,0,0],[255,0,255]])
   binary_codebook = np.array([[0],[1]])
   maskimage = recreate_image(binary_codebook, mask, w, h)
   maskimage = np.squeeze(maskimage)
   skimage.io.imsave ("/tmp/bw.jpg", maskimage)
   
   # Make a mask with that cluster
   # Make translation table 
   #################################################################################
   # Apply some morphops to that mask to ensure make the mask nicer and get rid of the noise
   import skimage.morphology
   selem = skimage.morphology.disk(6)
   selem2 = skimage.morphology.disk(1)
   
   #dilated = skimage.morphology.remove_small_objects(maskimage.astype("bool"), 10)
   #skimage.io.imsave ("/tmp/bw2.jpg", dilated.astype("float"))
   
   pipelined = maskimage.astype("bool").copy()
   #pipelined = skimage.morphology.erosion(pipelined.astype("bool"), selem)
   pipelined = skimage.morphology.erosion(pipelined.astype("bool"), selem)
   pipelined = skimage.morphology.remove_small_objects(pipelined.astype("bool"), 300)
   #pipelined = skimage.morphology.dilation(pipelined.astype("bool"), selem2)

   # Get the convex polygon for that mask
   pipelined = skimage.morphology.convex_hull_image(pipelined)
   fieldmask = pipelined.copy()
   skimage.io.imsave ("/tmp/fieldmask_convexhull.jpg", fieldmask.astype("float"))
   
   #################################################################################
   # Unifico los colores 
   import scipy.spatial.distance
   a = scipy.spatial.distance.cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)
   b = a.reshape((a.shape[0]*a.shape[1]))
   b[np.argsort(b)]
   
   #################################################################################
   # Mask the labels obtained before with me fieldmask, everything that is not within the fieldmask receives the mask of the background
   newcolors = kmeans.cluster_centers_.copy()
   #newcolors[maxlabel] = [0.,0.,0.]
   fieldlabels = labels.copy()
   fieldlabels[np.invert(fieldmask).ravel()] = maxlabel
   maskedfield = recreate_image(newcolors, fieldlabels, w, h)
   skimage.io.imsave ("/tmp/fieldmask.jpg", maskedfield.astype("float"))
   
   newcolors = kmeans.cluster_centers_.copy()
   newcolors[maxlabel] = [0.,0.,0.]
   fieldlabels = labels.copy()
   fieldlabels[np.invert(fieldmask).ravel()] = maxlabel
   maskedfield = recreate_image(newcolors, fieldlabels, w, h)
   skimage.io.imsave ("/tmp/fieldmask_blackbg.jpg", maskedfield.astype("float"))
   #################################################################################
   # Separo la imagen en las distintas labels
   labelmasks = map (lambda l: fieldlabels == l, np.unique(fieldlabels))
   
   # Identify colors corresponding to the team shirts
   redlab = skimage.color.rgb2lab ([[[ 0.68627451,  0.39215686,  0.47058824]]])
   yellowlab = skimage.color.rgb2lab ([[[ 0.5372549 ,  0.91764706,  0.2627451 ]]])
   
   lab_centers = skimage.color.rgb2lab ([kmeans.cluster_centers_])
   team1label = np.argmin(skimage.color.deltaE_cmc (lab_centers, redlab))
   team2label = np.argmin(skimage.color.deltaE_cmc (lab_centers, yellowlab))
   
   # Por cada label aplico un par de morphops
   # targetlabels = range(len(labelmasks))
   targetlabels = [ team1label-1, team2label-1 ]
   teammasks = []
   for i in  targetlabels :
      print i
      if i in maxlabels_list :
         continue
      labelmask = labelmasks[i].reshape(w,h)
      labelmask = np.squeeze(labelmask)
      #labelmask = np.invert(labelmask)
      #labelmask = skimage.morphology.binary_dilation(labelmask, selem)
      #labelmask = skimage.morphology.binary_erode(labelmask, selem)
      #labelmask = np.invert(labelmask)
      #labelmask = skimage.morphology.remove_small_objects(labelmask, 300)
      teammasks += [labelmask]
      skimage.io.imsave ("/tmp/labelmask_%i.jpg" % i, labelmask.astype("float"))
   
   return teammasks
   
   # Por cada contorno en cada label me quedo con los que "parecen jugadores" (ie son rectangulares y no gigantes)  (opcional)
   # Si tengo contornos solapados los unifico (opcional)
   # Make the mask that contains the field with that poly  (opcional)
      
   # From the kmeans clusters that I found before remove all pixels that that are not withing the convex polygon mask. After this step I should be able to make a mask this time with the blobs of the players
   
   #return maskedfield


# "/home/w1ndman/angelhack/frameproc/00000001.jpg"
if __name__ == "__main__" :
   parser = argparse.ArgumentParser(description='.')
   parser.add_argument('-i', '--input', help='Input image')
   parser.add_argument('-o', '--output', help='Output image')
   
   args = parser.parse_args()
   
   # infile = "/home/w1ndman/angelhack/frameproc/00000001.jpg"
   infile = args.input
   outfile = args.output
   
   print "Processing files"
   print "Input: %s" % infile
   print "Outfile: %s" % outfile
   
   # Load image
   imgdata = skimage.data.load(infile)
   getplayerblobs(imgdata)
   scalefact = 0.5
   getplayerblobs (imgdata, scalefact)


