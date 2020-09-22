import cv2, qrcode, os, time
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
import numpy as np
from camcom import Camcom
class QRreader:
  def __init__(self, option, data, save=False, plot=False):
    if option == 'write':
      self.im = [qrcode.make(d) for d in data]
      if save:
        [self.im[i].save(data[i]+'.png') for i in range(len(self.im))]

    if option == 'read':
      self.im, self.result = data, decode(data)
    
    if plot:
      plt.figure()
      plt.imshow(self.im)
      for r in self.result:
        if r.type == 'QRCODE':
          poly = r.polygon
          corners = np.array([[p.x, p.y] for p in poly])
          corners = np.concatenate((corners, np.reshape(corners[0, :], (1, 2))))
          plt.plot(corners[:, 0], corners[:, 1], 'o-')

if __name__ == '__main__':
  #cam = Camcom('settings/qr.pfs')
  #cam = Camcom('settings/acA2040_Default_cut.pfs')
  hits = 0
  for _ in range(1):
    im = cv2.imread('qr_test.png')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    qr = QRreader('read', im, plot=False)
    print(qr.result)
    if qr.result:
      hits += 1
  print('no hits:', hits)
  plt.imshow(im, cmap='gray')
  plt.title('no hits: '+str(hits))
  plt.show()
