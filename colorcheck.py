# -*- encoding: utf-8 -*-
'''
Author: Tomas Björnfot (tomas.bjornfot@ri.se)
Created: 2020-05-01
(c) copyright by RISE AB
'''
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import cv2, os, sys
import numpy as np
from hdpos import Hd_test, Hd
from pdb import set_trace as _stop

_arr = np.array
_join = os.path.join

def video_data(fname):
  '''
  Reads a video stream and convert each frame to a numpy array as
  (height, width, 6) with color order RGBHSV
  Parameters
  ----------
    fname : The image path
    NOTE: opencv use BRG and matplotlib (and others) uses RBG.
  Returns
  -------
    A list of numpy arrays as (height, width, 6) in RGBHSV
  '''
  data = []
  cap = cv2.VideoCapture(fname)
  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      data.append(frame_data(frame, 'BGR'))
    else:
      break
  return data

def frame_data(frame, order):
  '''
  Reads an image and outputs numpy arrays as (heigth, width, 6)
  with color order RGBHSV.
  Parameters
  ----------
    frame : The image frames as (height, width, 3)
    order : The color order of the input frame. It can be 'BGR' or
    'RGB'. 
    NOTE: opencv use BRG and matplotlib (and others) uses RBG.
  Returns
  -------
    A numpy array as (height, width, 6) in RGBHSV color order. 
  '''
  if order == 'BGR':
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  if order == 'RGB':
    return frame
  '''
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  return np.concatenate((rgb,hsv),axis=2)
  '''
def histogram_data(data):
  '''
  Collects histogram data from a list of frames.
  Parameters
  ----------
    data : A list of frames as numpy arrays as (height, width, 6)
    in RGBHSV color order.
  Returns:
  --------
    A list of 6 elements of [x, y]'s where x is the pixel value and y is
    the number of occurancies of each pixel value for all frames. Each 
    list element represents the color RGBHSV. 
  '''
  def flatten_frame(frame):
    '''
    Flattens a frame. I.e. (h, w, 6) => (h*w, 6)
    '''
    return _arr([frame[:, :, i].flatten() for i in range(3)])
  
  # flatten all frames
  x = _arr([flatten_frame(d) for d in data])
  # Merge each color to one array  
  data = [x[:, i, :].flatten() for i in range(3)]
  # getting the occurancies of each pixel value per color as a
  # precentage of the total 
  hist_data = []
  for d in data:
    w = np.ones(len(d)) / len(d)
    _, ax = plt.subplots()
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.close()
    z = ax.hist(d, 256, weights=w)
    z = [z[1][:-1], z[0]]
    ii = np.where(z[1] != 0)
    hist_data.append([z[0][ii], z[1][ii]])
  return hist_data

def plot_hist(bg, hd):
  '''
  A histogram plot of the video and the joint pixel values of all
  images in the folder. 
  '''
  _, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  _titles = ['Red', 'Green', 'Blue', 'Hue', 'Saturation', 'Value']
  for i in range(3):
    # --- Images with discs --- #
    _ax = ax[i]
    _ax.plot(hd[i][0], hd[i][1], color='C0', label='With disc')
    _ax.fill_between(hd[i][0], hd[i][1], color='C0', alpha=0.6)
    _ax.plot(bg[i][0], bg[i][1], color='C1', label='No disc')
    _ax.fill_between(bg[i][0], bg[i][1], color='C1', alpha=0.6)
    _ax.legend()
    _ax.set_title(_titles[i])
  plt.tight_layout()

def hist_from_files(_dir):
  '''
  Creates histograms from png or avi files
  Parameters
  ----------
    _dir : The dierctory where the files are.
  Returns
  -------
    - The plot object
  '''
  hd = []
  for name in os.listdir(_dir):
    path = _join(_dir, name)
    if name[-4:] == '.avi':
      bg = video_data(path)
    if name[-4:] == '.png':
      im = frame_data(cv2.imread(path), 'BGR')
      hd.append(im)
  hd_hist, bg_hist = histogram_data(hd), histogram_data(bg[:10])
  plot_hist(bg_hist, hd_hist)
  return plt

def carrier_check(mpath, spath):
  '''
  Checks the carrier for pixels outside the color definition. 
  Parameters
  ----------
    mpath : The path to the movie
    spath : The path to the settings file
  '''
  ims = video_data(mpath)
  hd, x = Hd(settingsfile=spath), []
  for im in ims:
    hd.img = im
    hd.bin_image()
    hd.blur()
    x.append([hd.bim, hd.bims])
    
  '''
  plt.figure()
  for i in range(30):
    plt.subplot(5, 6, i+1)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title('image number '+str(i))
    plt.imshow(x[i][0], cmap='gray')
  plt.show()
  '''

  for i, _x in enumerate(x):
    xsum = np.sum(_x[0])
    if xsum > 0:
      print('image nr:', i, end=' ')
      print(',pixels:', xsum)
  
if __name__ == '__main__':
  _dir = '../data/band1'
  sf = 'settings/settings_0813_verysmall_2.json'
  hist_from_files(_dir)
  carrier_check(_dir, sf)  
  x = Hd_test(sf).file_test(sf)
  for k in range(len(x)):
    plt.figure()
    for i in range(3):
      plt.subplot(1,3,i+1)
      plt.imshow(x[k].bims[i], cmap='gray')
  plt.show()
  print('done...')
