# -*- encoding: utf-8 -*-
'''
Author: Tomas Björnfot (tomas.bjornfot@ri.se)
(c) copyright by RISE AB
'''
import numpy as np
import cv2, os, json

_join = os.path.join
_arr = np.array

class Hd:
  '''
  Finds a hard drive in an image.
  It reads a settings 
  '''
  def __init__(self, im, settings_file=''):
    '''
    Loads settings and image.
    '''
    self.im = im
    if settings_file == '':
      self.settings = {
        "limits" : {
          "color": [[-1, 50], [-1, 50], [50, 100]],
          "hd_types": ["hd_35"], 
          "area":[[13500, 16000]]
          },
        "blur_level": 20,
        "ppmm": 6.55,
        "image_scale": 50,
        "outfile": '../.out',
        "result_dir": '../results',
        "wait": 0.5
      }
    else:
      with open(settings_file, 'r') as fp:
        self.settings = json.load(fp)
    self.ppmm = self.settings['ppmm']*self.settings['image_scale']/100

  def make_binary_image(self):
    '''
    Converts an image to a almost binary image.
    Reads settings/limits/color as:
      - [min, max] for each layer
    If the pixel in any layer lies outside any of the boundaries, 
    the binary pixel vill get the value +=1.
    Note that the settings/limits/color boundaries specifies
    the pixels in the background.
    '''
    im = self.im
    limits = self.settings['limits']['color']
    blevel = self.settings['blur_level']
    bim = np.zeros(im[:, :, 0].shape, np.uint8)
    for i in range(3):
      _lay, _lim = im[:, :, i], limits[i]
      _idx = np.where((_lay <= _lim[0]) | (_lay >= _lim[1]))
      bim[_idx] += 1
    bim = cv2.blur(bim, (blevel, blevel))
    bim[np.where(bim != 0)] = 1
    self.bim = bim

  def find_boxes(self):
    '''
    Calcuates contours surronding pixel 1 values. If the contour 
    area is within the requirement (settings/area), an enclosing
    box is put above the contour. The box(es) represents the hard
    drives.
    '''
    # takes the contours from the binary image
    ppmm = self.ppmm
    self.cnts, _ = cv2.findContours(self.bim, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    self.areas = _arr([cv2.contourArea(c) for c in self.cnts])/(ppmm*ppmm)
    # finds the contours that fits the area requirement
    ar = self.settings['limits']['area']
    va = []
    for _ar in ar:
      va.append(np.where((self.areas >= _ar[0]) & (self.areas <= _ar[1]))[0])
    self.valid_area_index = va
    # extracting the minimum box surronding the "area valid" contours  
    self.boxes, self.valid_area = [], []
    for i in self.valid_area_index:
      if len(i) > 0:
        self.valid_area.append(self.areas[i[0]])
        rect = cv2.minAreaRect(self.cnts[i[0]])
        self.boxes.append(cv2.boxPoints(rect))
  
  def get_box_dims(self):
    '''
    Calculates the dimensions of the boxes: area, width and length
    '''
    # getting the dims
    ppmm = self.ppmm
    self.length, self.width, self.theta = [], [], []
    for box in self.boxes:
      x = np.diff(box, axis=0)
      lx = np.sum(x**2, axis=1)
      i = np.argmax(lx)
      # getting the angle
      a = np.angle(x[i][0]+1j*x[i][1])*180/np.pi
      a = _arr([a+180, a, a-180])
      a = -a[np.where((a <= 90) & (a > -90))[0]]
      self.theta.append(a[0])
      # length and width
      self.length.append(np.sqrt(lx[i])/ppmm)
      self.width.append(np.sqrt(np.min(lx))/ppmm)
  
  def scale_image(self):
    '''
    Scales the image (im) as a procentage (_scale)
    '''
    _scale = self.settings['image_scale']
    if _scale != 100:
      w = int(self.im.shape[1] * _scale / 100)
      h = int(self.im.shape[0] * _scale / 100)
      self.im = cv2.resize(self.im, (w, h))

  def cut_out_hd(self, box_index=0):
    '''
    Cuts the image (im) around a box
    '''
    box = self.boxes[box_index]
    x = np.zeros(4, dtype=int)
    x[0] = int(np.min(box[:, 0]))
    x[1] = int(np.max(box[:, 0]))
    x[2] = int(np.min(box[:, 1]))
    x[3] = int(np.max(box[:, 1]))
    # the box can sometimes be outside the image
    x[np.where(x <= 1)[0]] = 1

    if x[1] > self.im.shape[1] - 1:
      x[1] = int(self.im.shape[1] - 1)
    
    if x[3] > self.im.shape[0] -1:
      x[3] = int(self.im.shape[3] -1)
    
    self.hd_im = self.im[x[2]:x[3], x[0]:x[1], :]

  def development_plot(self):
    '''
    Makes a plot using matplot lib. 
    Note that this function is only for development. Since it
    it's very slow. 
    '''
    import matplotlib.pyplot as plt
    plt.subplot(221)
    plt.title('original image')
    plt.imshow(self.im)
    plt.subplot(222)
    plt.title('binary image')
    plt.imshow(self.bim, cmap='gray')
    plt.subplot(223)
    plt.title('contours')
    plt.imshow(self.im)
    for c in self.cnts:
      _c = np.reshape(c, (c.shape[0], -1))
      plt.plot(_c[:, 0], _c[:, 1], color='C1')
    plt.subplot(224)
    plt.title('area valid contours with box')
    for box in self.boxes:
      box = np.int0(box)
      cv2.polylines(self.im, [box], True, (0, 255, 0), 4)

    plt.imshow(self.im)
    plt.tight_layout()
  
  def run(self):
    self.scale_image()
    self.make_binary_image()
    self.find_boxes()
    self.get_box_dims()
