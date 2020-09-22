'''
import cv2
import numpy as np
from qrreader import QRreader
# MAX 18 tecken
data = ['hejsan_svejsan__aa', 'tomas', 'björnfot33']
font = cv2.FONT_HERSHEY_SIMPLEX
qr = QRreader('write', data)
for i, _im in enumerate(qr.im):
  im = np.asarray(_im, np.uint8) 
  im[np.where(im == 1)] = 255 
  white_space = 255*np.ones((30, im.shape[1]))
  im = np.concatenate((im, white_space))
  cv2.putText(im, data[i], (10, im.shape[0]-20), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
  cv2.imwrite('test_'+str(i)+'.png', im)
'''
import qrcode, cv2, os
import numpy as np

_join = os.path.join
data = ['hejsan_svejsan', 'tomas', 'bj0rnfot33']
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(data)):
  qr = qrcode.QRCode(
      box_size=10,
      border=2,
  )
  qr.add_data(data[i])
  qr.make()

  _im = qr.make_image()
  im = np.asarray(_im, np.uint8) 
  im[np.where(im == 1)] = 255 
  white_space = 255*np.ones((30, im.shape[1]))
  im = np.concatenate((im, white_space))
  cv2.putText(im, data[i], (10, im.shape[0]-10), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
  fname = _join('result/qr_images', 'qr_'+data[i]+'.png')
  cv2.imwrite(fname, im)