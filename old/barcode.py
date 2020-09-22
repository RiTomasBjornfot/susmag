import matplotlib.pyplot as plt
import numpy as np
import zbar, cv2

fnames = ['barcode/barcode8_small.jpg']
for fname in fnames:
  
  # import image
  im = cv2.imread(fname, 0)

  # get bar data
  scanner = zbar.Scanner()
  result = scanner.scan(im)
  
  # plot
  plt.figure()
  plt.imshow(im, cmap='gray')
  if len(result) > 0:
    plt.title('type: '+result[0].type+' data: '+str(result[0].data.decode('utf-8')))
plt.show()
