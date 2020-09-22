import matplotlib.pyplot as plt
import numpy as np
import cv2
import pyflow, pickle, time, os
from pdb import set_trace as _stop

def read_image_sequence(_dir, layer):
  '''
  Reads all images in from _dir and sorts them according
  to end number in file name.
  Parameters
  ----------
    _dir : The directory where the sequence is.
  Returns
  -------
    A numpy array containing the images
  '''
  numbers, im = [], []
  for fname in os.listdir(_dir):
    if fname[-4:] == '.bmp':
      f = os.path.join(_dir, fname) 
      try:
        numbers.append(int(fname.split('.')[0].split('_')[-1]) - 1)
        im.append(plt.imread(f)[:, :, layer].astype(int))
      except:
        pass

  return [im[i] for i in np.argsort(numbers)]

def convert(im, rows, layer):
  '''
  Converts an image to smaller.
  NOTE: Not used!
  Parameters
  ----------
    im : The image to convert.
    rows : The as a list of two integers.
    layer : The image layer to use
  Returns
  -------
    An imge in one layer of intgers (0 - 255) with the same size as 
    im. 
  '''
  return im[rows[0]:rows[1], :, layer].astype(int)

def image_offset(im0, im1, width):
  '''
  Finding the optimal offset in the horizontal direction. The function
  sweeps a smaller version of im0 onto im1 and finds the minimum devation.
  NOTE: The function assumes that the objects in im0 has moved to the left 
  in im1. 
  Parameters
  ----------
    im0, im1 : The first and second image.
    width : The width (number of cols) in im1 to test
  Returns
  -------
    - A list of the pixel difference between the images 
    - The calculated pixel offset
  '''
  delta = []
  im0_small = im0[:, -width:]
  for i in range(1, 300):
    im1_calc = im1[:, (-i - width):-i]
    delta.append(np.sum(np.abs(im1_calc - im0_small)))
  return delta, np.argmin(delta)

if __name__ == '__main__':
  im = read_image_sequence('c:/dev/out', 2)
  width = 300
  layer = 2
  i= 12

  delta, offset = [], []
  #for i in range(len(im)-1):
  for i in range(10, 20):
    print(i, end=' ')
    print(np.std(im[i][:, -width:]))
    d, of = image_offset(im[i], im[i+1], width) 
    #d, of = image_offset(im[i], im[i+1], width) 
    delta.append(d)
    offset.append(of)
  

  [plt.plot(d, '.-', label=str(i)) for i, d in enumerate(delta)]
  plt.grid()
  plt.legend()
  plt.show()
  

  '''
  delta, offset = image_offset(im[i], im[i+1], width)
  plt.subplot(221)
  plt.imshow(im[i], cmap='gray')
  plt.title('im0')
  plt.subplot(222)
  plt.imshow(im[i+1], cmap='gray')
  plt.title('im1')
  plt.subplot(223)
  plt.title('calculated offset: '+str(offset))
  plt.plot(delta)
  plt.show()
  '''
  print('done...')
