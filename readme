=============================================
Paramters in the json settings file
=============================================
Image : (dict)
-----
  indir : (string) The directory where the images from the camera are stored
  outdir : (string) The directory where the resulting images and json data she, true, trueall be stored
  cut : (list of int) Where to cut the image files prior to calculation. sides and order 
    [vertical left, vertical right, horizontal top, horizontal bottom]
    Note that all pixel counting starts from top/left corner and index 
    wrapping can be used.

plot : (dict)
----
  save : (list of bool) Which plots that shall be saved (see plots.txt for details).
  sizes : (list of int)The figure size for all plots.
  dpi : The Dot Per Inch for all plots.

hdds : (dict) A set of harddrive sizes to judge its type
----
  name : (string) The name of the object.
  minw : (int) Minimum width.
  maxw : (int) Maximum width.
  minl : (int) Minimum length.
  maxl : (int) Maximum length.
  maxa : (int) The maximum allowed deviation in angulation of the corners (90 degrees == 0)
  maxr : (int) The maximum allowed resudual from the least square method

limits : (dict) Sets the thresholds for each layer for RGB AND HSV in the image. Each layer has its
------  own index according to order RGBHSV.
  indecies 0 - 6, where 0,1,2 is RGB and 3,4,5 is HSV. Note that the resulting binary
  image is an OR condition with respect to all limits. If a layer shall not be used,
  set upper "higher" than "lower".
  lower : (list of int) The lower limit for the disk for all layers.
  upper : (list of int) The upper limit for the disk for all layers.

render : (list of dict) This "cleans" the binary images from noise. A function looks a points in the 
------  convex hull (recursivly) and remove "1"'s in the binary image. Note that 
        renderer can contain many definitions.
  size : (int) The size of the square surrounding the point.
  density : (float) The minimum allowd density in the square.

=============================================
Message codes
=============================================
The message as "msg" is the sum of the following messages:
  + 1, wrong width
  + 2, wrong length
  + 4, corner angle deviation too large
  + 8, residual too large
If "msg" is -1, nothing was found in the image

=============================================
About Plotting files
=============================================

Control the writing of plots with the _plot parameter in test function as _plot = [True, True, True, True].
The images saved, (in the same order as the parameters)
1. Shows the final result. Saved as <infilename>_msg_<message>_result.png
2. Shows the meged binary image surrounded by the convex hull. To be used for development. 
  Saved as <infilename>_p1.png
3. Shows the image, convex hull, corners calculated from both least square
  method and directly from the convex hull. To be used for development. Saved as <infilename>_p2.png
4. Shows each binary image from each layer. To be used for development.  Saved as <infilename>_bims.png
