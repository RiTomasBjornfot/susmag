
import matplotlib.pyplot as plt
import numpy as np
import cv2, os
from camcom import Camcom
from hdpos import Hd
#remove this comment
_join = os.path.join

def run(iter, find_qr=False):
  # get image
  #im, _ = cam .grab_one()
  im = cv2.imread(_join(_dir, fname))
  im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
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

if __name__== '__main__':
  #cam = Camcom('settings/acA2040_Default.pfs')
  _dir = 'data/run'
  #while True:
  for i, fname in enumerate(os.listdir(_dir)):
    run(i, find_qr=False)