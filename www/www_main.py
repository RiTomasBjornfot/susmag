from flask import Flask, render_template, Response, redirect, url_for
from PIL import Image
import numpy as np
import json, os, cv2, time
from pdb import set_trace as _stop
from camcom import Camcom
from hdpos import Hd_test
from qrreader import QRreader

app = Flask(__name__)
_join = os.path.join

def gen_frames(info=False):
  cam = Camcom('../settings/acA2040_new.pfs')
  while True:
    _start = time.time()
    hdt = Hd_test('../settings/settings.json')
    im, _ = cam.grab_one()
    hdt.camera_test(im)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    frame = cv2.imencode('.jpg', im_rgb)[1].tobytes()

    if np.any([_msg == 0 for _msg in hdt.hd.msg]):
      print('found hard drive')
      # finding the qr_code
      im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      qr = QRreader('read', im_gray)
      try:
        qr_string = qr.result[0].data.encode()
      except:
        qr_string = str(round(time.time()))
      # saving the images
      img = Image.fromarray(im)
      img.save(_join('static/result', qr_string+'.png'))
      img.save(_join('static/show', 'show.png'))

    yield (b'--frame\r\n'            
          b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    if info:
      print('msg:', hdt.hd.msg, 'qr:', qr_string, 'time:', round(time.time() - _start, 3))

@app.route('/video_feed')
def video_feed():
  return Response(
    gen_frames(),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live(fname="static/test.png"):
  return render_template('live.html', fname=fname)

if __name__ == '__main__':
  #cam = start()
  app.run(debug=False)
