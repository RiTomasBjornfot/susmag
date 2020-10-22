import matplotlib.pyplot as plt
import numpy as np
import cv2, os,time
from camcom import Camcom
from hdpos import Hd
_join = os.path.join

def run(cam, iter, find_qr=False):
  # get image
  im, _ = cam.grab_one()
  # detect hd
  hd = Hd(im,settings_file='settings/settings.json')
  hd.run()
  if len(hd.valid_area) > 0:
    # hard drive coordinates
    # Note: only the first found hard drive
    box = np.int0(hd.boxes[0])
    # make and save image      
    img = cv2.cvtColor(hd.im, cv2.COLOR_BGR2RGB)
    cv2.polylines(img, [box], True, (0, 255, 0), 4)
    writename = str(iter) + '.png'
    cv2.imwrite(_join('result', writename), img)
  return hd

if __name__== '__main__':
  cam = Camcom('settings/acA2040_Default.pfs')
  i = 0
  while True:
    hd = run(cam, i)
    if len(hd.valid_area) > 0:
      i += 1
      time.sleep(0.5)
