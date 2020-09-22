import matplotlib.pyplot as plt
import numpy as np
import json, cv2
from pdb import set_trace as _stop

class Detect:
  def __init__(self, fname):
    '''
    Parameters
    ----------
      fname : the settings file (should be the same as the one used in hdpos)
    '''
    with open(fname, 'r') as fp:
      self.settings = json.load(fp)
  
  def is_carrier(self, im, col_index, limit):
    '''
    Checks if it's only carrier in the image.
    Parameters
    ----------
      im : The image an a numpy array of 6 layers
      col_index : A list of 2 values giving the start and end index of
      the coloums to use in the image.
      limit : The limit, as a precentage that is allowed to be outside
      the min, max values in the settings file.
    Returns
    -------
      1 if only the carrier is seen.
      0 if something else.
    '''
    part_outside = []
    for i in range(6):
      x = im[:, col_index[0]:col_index[1], i]
      _min , _max = self.settings['limits']['lower'][i], self.settings['limits']['upper'][i]
      if _min + _max == 0:
        part_outside.append(0)
        continue
      x_outside = x[(x < _min) | (x > _max)]
      part_outside.append(len(x_outside.flatten())/len(x.flatten()))
    for po in part_outside:
      if po > limit:
        return 0
    return 1
    
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

# --- MAIN --- #
if __name__ == '__main__':
  detect = Detect('settings/files.json')
  im = cv2.imread('data/band4/disc0_1.png')
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  im = detect.frame_data(im, 'RGB')

  print(detect.is_carrier(im, [0, 100], 0.1))