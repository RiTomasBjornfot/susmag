Control the writing of plots with the _plot parameter in test function as _plot = [True, True, True, True].
The images saved, (in the same order as the parameters)
1. Shows the final result. Saved as <infilename>_msg_<message>_result.png
2. Shows the meged binary image surrounded by the convex hull. To be used for development. 
  Saved as <infilename>_p1.png
3. Shows the image, convex hull, corners calculated from both least square
  method and directly from the convex hull. To be used for development. Saved as <infilename>_p2.png
4. Shows each binary image from each layer. To be used for development.  Saved as <infilename>_bims.png
