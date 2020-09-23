
print('\nMakes an installation test...')
success = 0
try:
  import pypylon
  print('pypylon:','OK', 'file', pypylon.__file__)
  success += 1
except:
  print('pypylon', 'ERROR')

try:
  import numpy
  print('numpy:','OK', 'version:', numpy.__version__)
  success += 1
except:
  print('numpy', 'ERROR')

try:
  import cv2
  print('opencv:','OK', 'version:', cv2.__version__)
  success += 1
except:
  print('opencv', 'ERROR')

if success == 3:
  print('Installation OK.\n')
else:
  print('Installation ERROR.\n')