# -*- encoding: utf-8 -*-
'''
Author: Tomas Björnfot (tomas.bjornfot@ri.se)
Created: 2020-05-01
(c) copyright by RISE AB
'''
import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace as _stop
from pypylon import pylon 
import time, os

class Camcom:
  '''
  A class to handle Basler cameras
  '''
  def __init__(self, configfile):
    self.cvt = self.converter()
    print('Connect:', end=' ')

    try:
      self.cam = self.connect()
      print('SUCCESS')
      print(self.cam.DeviceInfo)
    except Exception as e:
      print('FAILURE')
      print(e)

    print('Open:', end=' ')
    try:
      self.cam.Open()
      print('SUCCESS')
    except Exception as e:
      print('FAILURE')
      print(e)
      self.cam.close()
    
    print('Loading config from file :', configfile, end=' ')
    try:
      pylon.FeaturePersistence_Load(configfile, self.cam.GetNodeMap())
      print('SUCCESS')
    except Exception as e:
      print('FAILURE')
      print(e)
      
  def close(self):
    self.cam.Close()

  def connect(self):
    '''
    Connect to the first found pylon cam.
    NOTE: used by __init__
    Parameters
    ----------
      None
    Returns
    -------
      The cam object
    '''
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    return cam

  def converter(self):
    '''
    Setting up a converter to convert BGR to RGB
    NOTE: used by __init__
    Returns
    -------
      The converter
    '''
    cvt = pylon.ImageFormatConverter()
    cvt.OutputPixelFormat = pylon.PixelType_RGB8packed
    cvt.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return cvt

  def grab_one(self, info=False):
    '''
    Works better
    Returns
    -------
      - the image and the time is was taken
    '''
    if self.cam.IsOpen():
      self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
      while self.cam.IsGrabbing():
        res = self.cam.RetrieveResult(10, pylon.TimeoutHandling_Return)
        try:
          if res.GrabSucceeded():
            t0 = time.time()
            im = self.cvt.Convert(res).GetArray()
            self.cam.StopGrabbing()
            res.Release()
            return im, t0
        except Exception as e:
          if info: 
            print(e)
          pass

  def grab(self, no_images, fps, info=False):
    '''
    Grabs an image.
    Parameters
    -------
      no_images : The number of images to grab.
      fps : Number of frames per second.
      _dir : The directory to save the files
    Returns
    -------
      The image as an numpy array RGB image 
    '''
    if info:
      print('grab:', 'no_images=', no_images, 'fps=', fps)
    im = []
    for i in range(no_images):
      if info:
        print('grab image', i+1, end = ' ')
      if self.cam.IsOpen():
        if self.cam.IsGrabbing() == False:
          self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
        while self.cam.IsGrabbing():
          res = self.cam.RetrieveResult(200, pylon.TimeoutHandling_Return)
          if res.GrabSucceeded():
            im.append(self.cvt.Convert(res).GetArray())
            if info:
              print('SUCCESS')
            time.sleep(1/fps)
            break
          res.Release()
      else:
        print('No camera open')
        return np.array([])
    return im

  def grab_and_save(self, _dir, fname, no_images, fps):
    '''
    Grabs and saves images to file.
    Parameters
    ----------
      no_images : The number of images to grab.
      fps : Number of frames per second.
      _dir : The directory to save the files
      fname : The file name(s)
    '''
    print('grab_and_save:', '_dir=', _dir, 'fname=', fname, 'no_images=', no_images, 'fps', fps)
    im = self.grab(no_images, fps)

    for i in range(len(im)):
      _path = os.path.join(_dir, fname + '_'+ str(i+1) + '.png')
      plt.imsave(_path, im[i])

# --- MAIN --- #
if __name__ == '__main__':
  cam = Camcom('acA1300_2020-04-29.pfs')
  cam.grab_and_save('data/test', 'test.png', 1, 1)
  cam.close()
