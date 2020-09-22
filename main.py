import numpy as np
import time, cv2, json, os
from qrreader import QRreader
from camcom import Camcom
from hdpos import Hd_test

_join = os.path.join
_arr = np.array

def run(cam, hdt):

  _s = hdt.hd.settings['test']
  read_qr = _s['read_qr']
  show_video = _s['show_video']
  save_result = _s['save_result']
  max_iter = _s['max_iterations']
  delay = _s['delay_after_hit']
  save_all_images = _s['save_all_images']
  save_image_if_qr_fail = _s['save_image_if_qr_fail']

  if show_video:
    cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
    font = cv2.FONT_HERSHEY_SIMPLEX
  if save_result:
    result = {}
  if save_all_images or save_image_if_qr_fail:
    imgs=[]  

  result_index = 0
  while result_index < max_iter:
    im, gtime = cam.grab_one()
    _result = hdt.camera_test(im)
    qr_code = ''
    if read_qr:
      qr_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      qr = QRreader('read', qr_img)
      if len(qr.result) > 0:
        qr_code = qr.result[0].data.decode()

    if np.any(hdt.hd.msg) == 0:


      result_index += 1

      _result['qr'] = qr_code
      _result['epoch'] = gtime
      _result['datetime'] = time.ctime(gtime)
      yield _result

      if save_result:
        result[str(result_index)] = _result

      if show_video:
        p = [hdt.hd.lsqr_corners[i] for i in [0, 1, 3, 2]]
        p = [_arr(p, np.int32).reshape((-1,1,2))]
        img = cv2.cvtColor(hdt.hd.img, cv2.COLOR_BGR2RGB)
        cv2.polylines(img, p, True, (0, 255, 0), 1)
        for i, c in enumerate(hdt.hd.lsqr_corners):
          pos =(int(c[0]), int(c[1]))
          cv2.putText(img, str(i+1), pos, font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if read_qr:
          cv2.putText(img, qr_code, (350, 300), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('camera', img)
        cv2.waitKey(1)
        if save_all_images:
          imgs.append([result_index, img])
        if read_qr and qr_code == '' and save_image_if_qr_fail:
          imgs.append([result_index, img])
        time.sleep(delay)
    else: 
      if show_video:
        img = cv2.cvtColor(hdt.hd.img, cv2.COLOR_BGR2RGB)
        cv2.imshow('camera', img)
        cv2.waitKey(1)

    _t = round(time.time() - gtime, 2)
    print('time:', _t, 'message:', hdt.hd.msg, 'qr:', qr_code)

  if show_video:
    cv2.destroyAllWindows()
    
  if save_result:
    dte = time.ctime().replace(' ', '_').replace(':','-')
    fname = _join('result', dte+'.json')
    with open(fname, 'w') as fp:
      json.dump(result, fp, indent=2)

  if save_all_images or save_image_if_qr_fail:
    for img in imgs:
      fname = _join('result', str(img[0])+'.png') 
      cv2.imwrite(fname, img[1])

if __name__ == '__main__':
  cam_settings = 'settings/acA2040_Default.pfs'
  hd_settings = 'settings/settings.json'

  _cam = Camcom(cam_settings)
  _hdt = Hd_test(hd_settings)

  gen = run(_cam, _hdt)
  for _ in range(100):
    try:
      data = next(gen)
      print(data)
    except:
      break