qr_img = cv2.cvtColor(hd.hd_im, cv2.COLOR_BGR2GRAY)
qr = QRreader('read', qr_img)
  if len(qr.result) > 0:
    qr_code = qr.result[0].data.decode()
    qr_box = qr.result[0]
