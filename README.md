# SUSMAG
This is the repository for susmag release 1.

## Installation
Install the NI binaries from [here](https://www.ni.com/sv-se/support/downloads/software-products/download.labview.html#346254)
, the pylon camera suite from [here](https://www.baslerweb.com/en/sales-support/downloads/software-downloads/#type=pylonsoftware;version=all)
and Python 3 from [here](https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tar.xz).

Open a terminal and type the following:
```
pip install numpy pypylon python-opencv
```
Download a release from [here](www.google.com) and unzip the content to the c:\susmag directory.

Test the installation:
```
python c:\\susmag\\installation_test.py
```
## Calibration
Allways perform a calibration when the system has been moved.
### The position of the image
1. Start the pylon software and the light ramp.
2. Take a picture (F6).
3. Locate the position of the calibration nail in pixels.
4. Open the settings.json file.
5. In the **origin** key, type in the new value.
### The image scaling
1. Start the pylon software and the light ramp.
2. Add a ruler under the camera inline with the convoyer. 
3. Take a picture (F6).
4. Use the pylon viwer to estimate the number of pixels per milimeter.
5. Rotate the ruler 90 degrees and repeat the same action.
6. Check that the value is the same.
7. Open the settings.json file.
8. In the **ppmm** key, type in the new value.
### The position of the magnet sensor
1. Measure the relative distance along the convoyer between the nail and sensors.
2. Measure the relative distance perpendicular to the convoyer between the nail and closest sensor.
