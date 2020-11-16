# -*- encoding: utf-8 -*-
'''
Author: Tomas Björnfot (tomas.bjornfot@ri.se)
(c) copyright by RISE AB
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, time, json
from datetime import datetime as dt
from camcom import Camcom
from hdpos import Hd
from pdb import set_trace as _stop
_join = os.path.join

def signal_handler(fname):
  with open(fname, 'r') as fp:
    try:
      delta = float(fp.readline()) - time.time()
      print('delta:', delta)
      if delta < 0:
        return 0
      if delta > 0:
        time.sleep(delta)
        return 1
    except:
      print('cannot read signal file')
      return -1

def run(i, cam, settings):
  '''
  '''
  im, t0 = cam.grab_one()
  hd = Hd(im, settings)
  hd.run()
  fmt = hd.settings['image_format']
  imi = i % hd.settings['no_images']
  img = cv2.cvtColor(hd.im, cv2.COLOR_BGR2RGB)
  if len(hd.valid_area) > 0:
    print('harddrive detected', np.round(hd.valid_area[0]))
    box = np.int0(hd.boxes[0])
    cv2.polylines(img, [box], True, (0, 255, 0), 4)
    data = [i for item in hd.boxes[0] for i in item]
  else:
    print('no harddrive detected')
    data = [0 for item in range(8)]
  # write to files
  try:
    cv2.imwrite(_join(hd.settings['result_dir'], 'im_'+str(imi)+'.'+fmt), img)
    with open(hd.settings['outfile']+'_'+str(imi)+'.txt', 'w') as fp:
      fp.write(str(data)[1:-1].replace(', ', '\n')+'\n'+str(t0)+'\n')
  except:
    print('Can\'t write file!')
  # wait
  if len(hd.valid_area) > 0:
    time.sleep(hd.settings['waitdrive'])
  else:
    time.sleep(hd.settings['waitnodrive'])
  return i+1

if __name__== '__main__':
  # read the settings file
  fname = 'settings/settings.json'
  with open(fname, 'r') as fp:
    s = json.load(fp)
  # inits the camera
  cam = Camcom(s['camera_settings'])
  # The main loop
  i = 0
  while True:
    if signal_handler(s['signal_file']) == 1:
      i = run(i, cam, s)
