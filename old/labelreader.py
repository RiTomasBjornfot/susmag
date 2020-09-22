import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytesseract as pt
from PIL import Image
from pdb import set_trace as _stop

arr = np.array

class LabelReader:
  '''
  '''
  def __init__(self, im):
    self.im = im 
    self.imc = arr([])
    self.imb = arr([])
    self.center = arr([-1, -1])
    self.txt = []
    self.numbers = []
    self.offset = 40
    '''
    self.find_color_area()
    self.cut_image()
    self.find_text()
    self.text_to_int()
    '''
  def plot_label(self):
    plt.figure()
    plt.imshow(self.imc)
    plt.title(self.numbers)
    return plt
  def plot_result(self):
    '''
    Plots the result.
    '''
    plt.figure()
    plt.subplot(121)
    plt.title(str(self.center))
    plt.imshow(self.im, cmap='gray')
    plt.subplot(122)
    plt.title(self.numbers)
    plt.imshow(self.imc, cmap='gray')
    plt.tight_layout()
    return plt

  def find_color_area(self):
    '''
    Detects the biggest color area in the image
    '''
    im = self.im
    i = np.where((im[:,:,0]>200) & (im[:,:,1]<200) & (im[:,:,2]<200))
    imb = np.zeros(im.shape[:2])
    imb[i] = 128
    imb = cv2.GaussianBlur(imb, (3, 3), 0)
    i = np.where(imb<125)
    imb[i] = 0
    i = np.where(imb>=125)
    imb[i] = 128
    self.imb = imb

    try:
      ctr = np.median(i, axis=1)
      self.center = [int(c) for c in ctr]
    except Exception as e:
      print(e)

  def cut_image(self):
    '''
    Cuts out a smaller image from an image.
    '''
    of, cc, im = self.offset, self.center, self.im
    self.imc = im[(cc[0]-of):(cc[0]+of), (cc[1]-of):(cc[1]+of) , :]

  def find_text(self):
    '''
    Finds a text in an image.
    '''
    self.txt = []
    for k in range(4):
      _im = np.rot90(self.imc, k=k)
      self.txt.append(pt.image_to_string(_im))

  def text_to_int(self):
    '''
    Convers a text list to numbers.
    '''
    self.numbers = []
    for _txt in self.txt:
      try:
        self.numbers.append(int(_txt))
      except:
        pass

  def make_small_binary(self):
    '''
    makes a binary image to make the txt more clear.
    NOTE: Not used because it seems to work anyway.
    '''
    imc = self.imc
    imcb = np.ones(imc.shape[:2])
    i = np.where((imc[:,:,1] > 150) & (imc[:,:,2] > 90))
    imcb[i] = 0
    self.imcb = imcb