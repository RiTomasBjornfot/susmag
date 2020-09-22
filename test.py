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
import matplotlib.pyplot as plt

_join = os.path.join
data = ['company_'+str(i) for i in range(65)]
font = cv2.FONT_HERSHEY_SIMPLEX
ratio = 1.7971698113207548
width = 250
length = (width*ratio)
_ws_length = int(length - width)
h_border = 126
v_border = 126
cols = []
for col in range(5): 
  for row in range(13):
    qr = qrcode.QRCode(
        box_size=10,
        border=2,
    )
    index = 13*col+row
    qr.add_data(data[index])
    qr.make()

    _im = qr.make_image()
    im = np.asarray(_im, np.uint8) 
    im[np.where(im == 1)] = 255 
    white_space = 255*np.ones((_ws_length, im.shape[1]))
    im = np.concatenate((im, white_space))
    cv2.putText(im, data[index], (10, im.shape[0]-_ws_length + 10), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    im = np.rot90(im)
    if row == 0:
      _col = im
    else:
      _col = np.concatenate((_col, im))
  cols.append(_col)

mat = cols[0]
for col in cols[1:]:
  mat = np.concatenate((mat, col), axis=1)

img = 255*np.ones((2*h_border + mat.shape[0], 2*v_border + mat.shape[1]))
img[h_border:-h_border, v_border:-v_border] = mat

fname = _join('result/qr_images', 'qr_codes'+'.png')
cv2.imwrite(fname, img)