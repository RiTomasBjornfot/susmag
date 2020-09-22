
import matplotlib.pyplot as plt
import numpy as np
import cv2, os
from camcom import Camcom
from hdpos2 import Hd
from qrreader import QRreader

_join = os.path.join
def run(iter, find_qr=False):
  # get image
  #im, _ = cam .grab_one()
  im = cv2.imread(_join(_dir, fname))
  im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
  # detect hd
  hd = Hd(im)
  hd.run()
  if len(hd.valid_area) > 0:
    # hard drive coordinates
    # Note: only the first found hard drive
    box = np.int0(hd.boxes[0])
    # find qr code
    if find_qr:
      qr_code = 'noqr'
      hd.cut_out_hd()
      qr_img = cv2.cvtColor(hd.hd_im, cv2.COLOR_BGR2GRAY)
      qr = QRreader('read', qr_img)
      if len(qr.result) > 0:
        qr_code = qr.result[0].data.decode()
        qr_box = qr.result[0]
    # make and save image      
    img = cv2.cvtColor(hd.im, cv2.COLOR_BGR2RGB)
    cv2.polylines(img, [box], True, (0, 255, 0), 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, qr_code, (20, hd.im.shape[0]-20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    writename = str(iter) + '_' + qr_code  + '.png'
    cv2.imwrite(_join('result', writename), img)

if __name__== '__main__':
  #cam = Camcom('settings/acA2040_new_lens.pfs')
  _dir = 'data/run'
  #while True:
  for i, fname in enumerate(os.listdir(_dir)):
    run(i, find_qr=True)