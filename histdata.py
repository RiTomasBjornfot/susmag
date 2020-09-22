import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import cv2, os, sys, pickle
import numpy as np
from pdb import set_trace as _stop

_arr = np.array
_join = os.path.join

class HistData:
  '''
  A class to get the hist data to from:
    - A video recording of the background
    - A set of images of discs
  Use from_files() if the data is stored on files
  Use from_data() if data is local
  '''
  def __init__(self):
    pass

  def bandcheck_from_files(self, dir_, _plot=True):
    '''
    Add comment
    '''
    plt.close('all')
    hd = []
    for name in os.listdir(dir_):
      path = _join(dir_, name)
      if name[-4:] == '.pic':
        print('Reading file', path, 'as background.')
        bg = pickle.load(open(path, 'rb'))
      if name[-4:] == '.png':
        print('Reading file', path, 'as disk.')
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im); plt.show()
        hd.append(im)
    self.bandcheck(bg, hd, _num ='BANDCHECK FROM FILES')
  
  def bandcheck(self, bg, hd, _plot=True, _num=''):
    '''
    Add comment
    '''
    bg = [self.frame_data(_bg, 'RGB') for _bg in bg]
    self.bg_hist = self.histogram_data(bg)
    hd = [self.frame_data(_hd, 'RGB') for _hd in hd]
    self.hd_hist = self.histogram_data(hd)
    if _plot:
      self.plot_hist(_num='')

  def video_data(self, fname):
    '''
    Reads a video stream and convert each frame to a numpy array as
    (height, width, 6) with color order RGBHSV
    Parameters
    ----------
      fname : The image path
      NOTE: opencv use BGR and matplotlib (and others) uses RGB.
    Returns
    -------
      A list of numpy arrays as (height, width, 6) in RGBHSV
    '''
    data = []
    cap = cv2.VideoCapture(fname)
    while cap.isOpened():
      ret, frame = cap.read()
      if ret:
        data.append(self.frame_data(frame, 'BGR'))
      else:
        break
    return data

  def frame_data(self, frame, order):
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
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if order == 'RGB':
      rgb = frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return np.concatenate((rgb,hsv),axis=2)

  def histogram_data(self, data):
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
      return _arr([frame[:, :, i].flatten() for i in range(6)])
    
    # flatten all frames
    x = _arr([flatten_frame(d) for d in data])
    # Merge each color to one array  
    data = [x[:, i, :].flatten() for i in range(6)]
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

  def plot_hist(self, _num):
    '''
    A histogram plot of the video and the joint pixel values of all
    images in the folder. 
    '''
    _, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 5), num=_num)
    _titles = ['Red', 'Green', 'Blue', 'Hue', 'Saturation', 'Value']
    bg, hd = self.bg_hist, self.hd_hist
    for i in range(6):
      # --- Images with discs --- #
      _ax = ax[i //3, i % 3]
      _ax.plot(hd[i][0], hd[i][1], color='C0', label='With disc')
      _ax.fill_between(hd[i][0], hd[i][1], color='C0', alpha=0.6)
      _ax.plot(bg[i][0], bg[i][1], color='C1', label='No disc')
      _ax.fill_between(bg[i][0], bg[i][1], color='C1', alpha=0.6)
      _ax.legend()
      _ax.set_title(_titles[i])
    plt.tight_layout()
    plt.show()

