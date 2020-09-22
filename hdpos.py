# -*- encoding: utf-8 -*-
'''
Author: Tomas Björnfot (tomas.bjornfot@ri.se)
Created: 2020-05-01
(c) copyright by RISE AB
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, pickle, json, random, time
from scipy.interpolate import interp1d
from numpy.linalg import lstsq
from pdb import set_trace as _stop

rs = np.reshape
arr = np.array
join =  os.path.join

class Hd:
  def __init__(self, settingsfile='settings.json'):

    self.settings = json.load(open(settingsfile, 'r'))
    
    self.name = ''
    self.img = arr([])
    self.bims = [] 
    self.bim = arr([])
    self.hull = arr([])

    self.props = []
    self.angle = 0 
    self.width = 0
    self.length = 0
    self.hull_corners = arr([])
    self.lsqr_corners = arr([])
    self.corner_angles = arr([])
    self.residual = []
    self.msg = [-1]
    self.error = ''

  '''
  ============== IMAGE AND HULL  ================ 
  '''
  def cutimage(self, im):
    cut, indir = self.settings['image']['cut'], self.settings['image']['indir']
    self.img = im[cut[0]:cut[1], cut[2]:cut[3], :]

  def scaleimage(self):
    '''
    Scales self.img.
    '''
    scale = self.settings['image']['scale']
    width = int(self.img.shape[1] * scale / 100)
    height = int(self.img.shape[0] * scale / 100)
    dim = (width, height)
    self.img = cv2.resize(self.img, dim, interpolation = cv2.INTER_AREA)

  def loadimage(self):
    cut, indir = self.settings['image']['cut'], self.settings['image']['indir']
    frame = cv2.imread(join(indir, self.name))[cut[0]:cut[1], cut[2]:cut[3], :]
    self.img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  def bin_image(self):
    '''
    Creates an image by extracting out the background. The limits are set 
    according to the carrier. Thus, if the carrier pixels are between 100 and 200
    for a layer the min should be 100 and max 200 for that layer.
    '''
    _min = self.settings['limits']['lower']
    _max = self.settings['limits']['upper']
    _bim = np.zeros(self.img[:, :, 0].shape)
    # bgr
    ts = np.zeros(self.img[:, :, 0].shape, np.uint8)
    self.bims = []
    for i in range(3):
      _im = np.copy(ts)
      if _min[i] + _max[i] != 0:
        im = self.img[:, :, i]
        idx = np.where((im <= _min[i]) | (im >= _max[i]))
        _im[idx] = 1
      self.bims.append(_im)
      _bim += _im

    # hsv
    '''
    hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
    for i in range(3):
      _im = np.copy(ts)
      if _min[i+3] + _max[i+3] != 0:
        im = hsv[:, :, i]
        idx = np.where((im <= _min[i+3]) | (im >= _max[i+3]))
        _im[idx] = 1
      self.bims.append(_im)
      _bim += _im
    
    idx = np.where(_bim > 0)
    _bim[idx] = 255
    '''
    self.bim = _bim

  def blur(self):
    '''
    Bluring the image to remove small pixels.
    '''
    for bval in self.settings['blur']:
      self.bim = np.round(cv2.blur(self.bim , (bval, bval)))

  def remove_clusters(self, img, center, size, min_density):
    '''
    Takes a part of an image and check the density. If the density is
    lower than min_density then all pixels in the region is set to 0.
    '''
    try:
      x0, x1 = center[0]-size, center[0]+size
      y0, y1 = center[1]-size, center[1]+size

      if x0 <= -1:
        x0 = 0
      if y0 <= -1:
        y0 = 0
    
      if x1 >= img.shape[1] - 1:
        x1 = img.shape[1] - 1
      if y1 >= img.shape[0] - 1:
        y1 = img.shape[0] - 1

      _img = img[y0:(y1+1), x0:(x1+1)].astype('uint32')
      density = np.sum(_img)/(((y1 - y0)*(x1 - x0))*255)
      action = 0
      if density < min_density:
        img[y0:(y1+1), x0:(x1+1)] = np.zeros((y1-y0+1, x1-x0+1))  
        #print('x:', center[0], 'y:', center[1], 'density:', density)
        action = 1
    except Exception as e:
      print('ERROR in remove_clusters()', e)
    return img, action

  def render(self):
    '''
    Render the image by removing small clusters in the image at
    the hull points.
    '''
    self.get_hull()
    action = 1
    while action != 0:
      action = 0
      for i in range(self.hull.shape[0]):
        for s in self.settings['render']:
          self.bim, a = self.remove_clusters(self.bim, self.hull[i, :], s['size'], s['density'])
          action += a
      self.get_hull()

  def get_hull(self):
    '''
    Gets the convex hull from a binary image
    '''
    p = arr(np.where(self.bim.T == 1)).T
    h = cv2.convexHull(p)
    h = rs(cv2.convexHull(p), (h.shape[0], h.shape[2]))
    h = np.concatenate((h, rs(h[0, :], (1, 2))), axis=0)
    self.hull = h

  '''
  ============== RECTANGLE PROPERTIES ================ 
  '''

  def get_props(self):
    hull = self.hull
    move = np.mean(hull, axis=0)
    # gets the angle
    _rs = np.arange(-90, 90, .2)
    hulls = [self.rot(hull - move, r) for r in _rs]
    dy = [np.max(h[: ,1]) - np.min(h[:, 1]) for h in hulls]
    i = np.argmin(dy)
    # gets the angle
    angle = _rs[i]
    # gets the width
    w = dy[i]
    # gets the length
    l = np.max(hulls[i][:, 0]) - np.min(hulls[i][:, 0])
    # get the corners
    h = self.rot(hull - move, 30 + angle)
    amax, amin, x, y = np.argmax, np.argmin, h[:, 0], h[:, 1]
    idx = [amax(y), amax(x), amin(x), amin(y)]
    corner = arr([hull[i, :] for i in idx])
    # getting least square corners
    lc, ca, res = self.get_lsqr_corners(hull, angle, corner)
    # add to instance
    self.angle = angle
    self.width = w
    self.length = l
    self.hull_corners = corner
    self.lsqr_corners = lc
    self.corner_angles = ca
    self.residual = res

  '''
  Least square stuff
  '''
  def rot(self, pts, degree):
    '''
    Rotates a set of nX2 data in 2D
    '''
    theta = np.radians(degree)
    _c, _s = np.cos(theta), np.sin(theta)
    rmat = arr(((_c,-_s), (_s, _c)))
    return np.dot(rmat, pts.T).T

  def get_sides(self, hull, corners):
    '''
    Gets the sides of a rectangle. The resctange must be aligned with 
    the long side along the x axis. 
    '''
    h, c = hull, corners
    # get the sides
    ca = [np.angle(c[i, 0] + 1j*c[i, 1])  for i in range(c.shape[0])]
    sides = [[], [], [], []]
    for i in range(h.shape[0]):
      a = np.angle(h[i, 0] + 1j*h[i, 1]) 
      # right
      if a <= ca[0] and a >= ca[1]:
        sides[0].append(h[i, :])
      # top
      if a <= ca[2] and a >= ca[0]:
        sides[1].append(h[i, :])
      # left
      if a <= ca[3] or a >= ca[2]:
        sides[2].append(h[i, :])
      # bottom
      if a <= ca[1] and a >= ca[3]:
        sides[3].append(h[i, :])
    return [arr(s) for s in sides] 

  '''
  lsqr corners
  '''
  def interpolate(self, side, no_pts):
    '''
    Interpolates on a line. Used by lsqr to add points to the sides.
    Used only by get_lsqr_corners()
    '''
    f = interp1d(side[:, 0], side[:, 1])
    xn = np.linspace(np.min(side[:, 0]), np.max(side[:, 0]), no_pts)
    return arr([xn, f(xn)]).T

  def line_crossing(self, line1, line2):
    '''
    Finds the crossing of two lines. Used only by get_lsqr_corners() 
    '''
    x = (line1[0] - line2[0])/(line2[1] - line1[1])
    y = line1[1]*x + line1[0]
    return [x, y]

  def calc_corner_angle(self, p0, p1, p2):
    '''
    Finds the angle between points as corners. Used only by 
    get_lsqr_corners().
    '''
    v1, v2 = p1 - p0, p2 - p0
    a1 = np.angle(v1[0] + 1j*v1[1])
    a2 = np.angle(v2[0] + 1j*v2[1])
    angle = (a1 - a2)*180/np.pi
    if angle < -180.0:
      angle += 360
    return angle


  def remove_points_near_corners(self, pts, precent):
    '''
    Removes posits near corners. Used to make a more stable 
    lsqr solution. Points in sides near corners tend to be
    more noisy. Used only by get_lsqr_corners()
    '''
    _min = np.min(pts[:, 0])
    _max = np.max(pts[:, 0])
    _min += precent*(_max - _min)
    _max -= precent*(_max - _min)
    npts = []
    for i in range(pts.shape[0]):
      if pts[i, 0] > _min and pts[i, 0] < _max:
        npts.append([pts[i, 0], pts[i, 1]])
    return arr(npts)

  def get_lsqr_corners(self, hull, angle, corners):
    move = np.mean(hull, axis=0)
    h = self.rot(hull - move, angle)[:-1, :]
    c = self.rot(corners - move, angle)
    hull_sides, sides, lsqdata = self.get_sides(h, c), [], []

    for i, s in enumerate(hull_sides):
      # rotate
      s = self.rot(s, 45)
      # interpolate
      s = self.interpolate(s, 100)
      # remove points near corners
      s = self.remove_points_near_corners(s, 0.2)
      #lsqr
      a = np.ones(s.shape)
      a[:, 1] = s[:, 0]
      b = rs(s[:, 1], (len(s[:, 1]), 1)) 
      sides.append(s)
      lsqdata.append(lstsq(a, b, rcond=None))

    
    lines = [rs(arr(q[0]), -1) for q in lsqdata] 
    lc, ls = self.line_crossing, lines
    # calculate new corners using line crossing
    corners = arr([lc(ls[0],ls[1]), lc(ls[0],ls[3]), 
        lc(ls[1],ls[2]), lc(ls[2],ls[3])])
    # calculate angle in the corners
    c, ca = corners, self.calc_corner_angle
    corner_angle = arr([ca(c[0, :],c[1, :], c[2, :]), ca(c[1, :],c[3, :],c[0, :]), 
      ca(c[2, :], c[0, :],c[3, :]), ca(c[3, :], c[2, :],c[1, :])])

    corners = self.rot(corners, -(45 + angle)) + move
    residuals = [float(lsq[1]) for lsq in lsqdata]
    return  corners, corner_angle, residuals

  '''
  other stuff
  '''

  def get_msg(self):
    '''
    Calculates the result message.
    '''
    self.msg = []
    mmpp = 1.0/self.settings['dims']['ppmm']
    self.width *= mmpp
    self.length *= mmpp

    for s in self.settings['dims']['types']:
      try:
        msg = 0
        if self.width < s['minw'] or self.width > s['maxw']:
          msg += 1
        if self.length < s['minl'] or self.length > s['maxl']:
          msg += 2
        if (self.corner_angles < (90 - s['maxa'])).any() or (self.corner_angles > (90 + s['maxa'])).any():
          msg += 4
        if (arr(self.residual ) > s['maxr']).any():
          msg += 8
      except:
        msg = -1
      self.msg.append(msg)

  def to_dict(self, result, to_file=True):
    '''
    Writes the result to a dictionary
    '''
    def to_2d_list(x):
      return [list(_x) for _x in list(x)]

    d = {}
    for i, _msg in enumerate(self.msg):
      if _msg == 0:
        dx = self.settings['image']['cut'][0]
        dy = self.settings['image']['cut'][2]
        ppmm = self.settings['dims']['ppmm']
        self.org_corners = np.round(self.lsqr_corners, 2)
        self.org_corners[:, 0] += dx
        self.org_corners[:, 1] += dy
        _type = self.settings['dims']['types'][i]['name']
        d[_type] = {}
        d[_type]['corners'] = to_2d_list(self.org_corners)
        d[_type]['angle'] = round(self.angle, 2) 
        d[_type]['width'] = round(self.width, 1)
        d[_type]['length'] = round(self.length, 1)
        d[_type]['residual'] = round(np.max(self.residual), 2)
        d[_type]['angledeviation'] = round(np.max(np.abs(self.corner_angles - 90)), 2)
        d[_type]['msg'] = _msg
    return d

  '''
  =============== PLOTTING ================
  '''
  def devplot1(self):
    plt.figure(figsize=self.settings['plot']['size'])
    plt.imshow(self.bim, cmap='gray')
    if self.hull.shape[0] > 0:
      plt.plot(self.hull[:, 0], self.hull[:, 1], '.-', color='C0')
    plt.axis('off')
    outdir = self.settings['image']['outdir']
    dpi = self.settings['plot']['dpi']
    plt.savefig(join(outdir, self.name+'_p1.png'), dpi=dpi)
    plt.close()

  def devplot2(self):
    plt.figure(figsize=self.settings['plot']['size'])
    plt.imshow(self.img, cmap='gray')
    if len(self.hull) != 0:
      plt.plot(self.hull[:, 0], self.hull[:, 1], 
          '.-', color='C0', label='hull')
      if self.msg[0] != -1:
        plt.plot(self.lsqr_corners[:, 0], self.lsqr_corners[:, 1], 
            'o', color='C2', label='lsqr corners')
        plt.plot(self.hull_corners[:, 0], self.hull_corners[:, 1], 
            'o', color='C3', label='hull corners')
        plt.legend()
    plt.title(self.name)
    plt.tight_layout()
    plt.axis('off')
    outdir = self.settings['image']['outdir']
    dpi = self.settings['plot']['dpi']
    plt.savefig(join(outdir, self.name + '_msg_'+str(self.msg[0])+'_p2.png'), dpi=dpi)
    plt.close()

  def resultplot(self, random_name=False, arrow=False):
    plt.figure(figsize=self.settings['plot']['size'])
    plt.imshow(self.img)
    cns = self.lsqr_corners
    
    if cns.shape[0] > 0:
      pts = arr([cns[0, :], cns[1, :], cns[3, :], cns[2, :], cns[0, :]])
      plt.plot(pts[:, 0], pts[:, 1], 'o-', color='C3')
      if arrow:
        center = np.mean(cns, axis=0)
        plt.plot(center[0], center[1], 'o', color='C3')
        dx, dy = np.cos(self.angle*np.pi/180.0), -np.sin(self.angle*np.pi/180.0)
        plt.arrow(center[0], center[1], 200*dx, 200*dy, color='C3', head_width=30)
    
    plt.axis('off')
    outdir = self.settings['image']['outdir']
    dpi = self.settings['plot']['dpi']
    if random_name == False:
      savename = self.name + '_msg_'+str(self.msg[0])+'_result.png'
    if random_name == True:
      nr = str(int(random.random()*1000000))
      savename = self.name + '_' + nr + '_msg_'+str(self.msg[0])+'_result.png'
    plt.savefig(join(outdir, savename), dpi=dpi)
    plt.close()
  
  def bimsplot(self):
    plt.figure(figsize=self.settings['plot']['size'])
    for i in range(3):
      plt.subplot(2,3,i+1)
      plt.title('layer no: '+str(i))
      plt.imshow(self.bims[i], cmap='gray')
    fname = join(self.settings['image']['outdir'], self.name+'_bims.png')
    plt.savefig(fname)
    plt.close()

  def rawplot(self):
    plt.figure(figsize=self.settings['plot']['size'])
    plt.imshow(self.img)
    fname = join(self.settings['image']['outdir'], self.name+'_raw.png')
    plt.savefig(fname)
  '''
  ============= PUBLIC FUNCTIONS ================
  '''
  def find(self):
    self.bin_image()
    self.blur()
    self.get_hull()
    #self.render()
    if self.hull.shape[0] > 0:
      self.get_props()
      self.get_msg()
    else:
      self.error = 'No hull found'

  '''
  ============= TEST CLASS ================
  '''
class Hd_test:

  def __init__(self, settingsfile):
    self.hd = Hd(settingsfile=settingsfile)
  
  def plot(self, random_name=False):
    _plot = self.hd.settings['plot']['save']
    if _plot[0]:
      self.hd.resultplot(random_name=random_name)
    if _plot[1]:
      self.hd.devplot1()
    if _plot[2]:
      self.hd.devplot2()
    if _plot[3]:
      self.hd.bimsplot()
    if _plot[4]:
      self.hd.rawplot()

  def camera_test(self, im, random_name=False, info=False):
    '''
    Add comments...
    '''
    self.hd.cutimage(im)    
    self.hd.scaleimage()
    self.hd.name = 'test_'+str(round(time.time()))[-4:]    
    try:
      self.hd.find()
    except Exception as e:
      self.hd.msg = [-1]
    if info:
      try:
        print('message:', str(self.hd.msg), self.hd.error, 
          'residual:', np.round(np.max(self.hd.residual), 2))
      except:
        print('message:', str(self.hd.msg), self.hd.error)
    return self.hd.to_dict(self.hd)

  def file_test(self, sf):
    '''
    Add comments...
    NEEDS TO BE FIXED. Look at camer_test
    '''
    hds, _dir = [], Hd(settingsfile=sf).settings['image']['indir']
    for fname in os.listdir(_dir):
      if fname[-4:] == '.bmp':
        hd = Hd(settingsfile=sf)
        hd.name = fname
        hd.loadimage()
        try:
          hd.find()
        except Exception as e:
          hd.msg = [-1]
          hd.error = str(e)
        hds.append(hd)
        try:
          print(fname, ', message:', str(hd.msg), hd.error, 
            'residual:', np.round(np.max(hd.residual), 2))
        except:
          print(fname, ', message:', str(hd.msg), hd.error)
        _plot = hd.settings['plot']['save']
        if _plot[0]:
          hd.resultplot()
        if _plot[1] and hd.msg[0] != 0:
          hd.devplot1()
        if _plot[2] and hd.msg[0] != 0:
          hd.devplot2()
        if _plot[3] and hd.msg[0] != 0:
          hd.bimsplot()
        hd.to_dict(hd)
    return hds
