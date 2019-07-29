

Author: Nicolas Blondel - Computer Vision Lab, ETH Zürich
Created: 29.07.2019
==========================================================

Random Walker Based 4D Flow MRI Segmentation Tool

==========================================================

Created and tested with Python XX.XX and Tensorflow xx.xx

Required extensions:

mayavi
tkinter
PIL


---------------------------------------------------------
Random Walker GUI

Main file to run: 	random_walker_gui.py
Requires: 		functions.py, random_walker_3D.py, random_walker_4D.py

Workflow for best segmentation result:

1) Cycle through display modes ("Overlab Image"-Button) until the image with the velocity magnitude is shown
2) Explore the 4D volume with the sliders "t-axis" and "z-axis" until you see good contrast in the velocity
3) Press "Scribble FG" to add a scribble to the foregound, draw a couple of lines inside the aorta
4) Press "Scribble BG" and repeat
5) T and z axis are frozen (on purpose)
6) Press "Add scribble" to unfreeze and add more scribbles in other slices
7) Add 2-3 more scribbles at other slices ("z-axis"-slider)
8) Press "Run 3D" to run the algorithm in 3D
9) To propagate the 3D result to a 4D segmentation press "Load segm -> markers" (uses 3D segmentation as rough markers for other timesteps with dilation and erosion)
10) Press "Run 4D" to run the algorithm in 4D using the precomputed markers from the 3D segmentation
11) Done!

To reset and rerun the RW simply press "Quit" and restart the program

---------------------------------------------------------

