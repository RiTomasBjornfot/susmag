# -*- encoding: utf-8 -*-
'''
Author: Tomas Björnfot (tomas.bjornfot@ri.se)
(c) copyright by RISE AB
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, time
from datetime import datetime as dt
from camcom import Camcom
from hdpos import Hd

_join = os.path.join

def run(i, cam):
  t0, im = time.time(), cam.grab_one()
  hd = Hd(im,settings_file='settings/settings.json')
  hd.run()
  if len(hd.valid_area) > 0:
    box = np.int0(hd.boxes[0])
    img = cv2.cvtColor(hd.im, cv2.COLOR_BGR2RGB)
    cv2.polylines(img, [box], True, (0, 255, 0), 4)
    fname = dt.now().strftime('%Y%m%d_%H%M%S')+str(i)+'.png'
    cv2.imwrite(_join(hd.settings['result_dir'], fname), img)
    with open(hd.settings['outfile'], 'w') as fp:
      data = [i for item in hd.boxes[0] for i in item]
      fp.write(str(data)[1:-1]+', '+str(t0)+'\n')
    time.sleep(hd.settings['wait'])
    i += 1
  return i

if __name__== '__main__':
  cam, i = Camcom('settings/acA2040_NoCrop.pfs'), 0
  while True:
    i = run(i, cam)
