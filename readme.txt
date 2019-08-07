

Author: Nicolas Blondel - Computer Vision Lab, ETH Zürich
Created: 29.07.2019
Last updated: 07.08.2019
==========================================================

Random Walker Based 4D Flow MRI Segmentation Tool

Version 0.0.2

Latest updates:

- Scribbles are always visualized after drawing them
- Possibility for choice to run algorithm in 3D or 4D
- Button to view segmentation overlap on original image

==========================================================

Created and tested with Python XX.XX and Tensorflow xx.xx

Required extensions:

mayavi
tkinter
PIL


---------------------------------------------------------
Random Walker GUI

Main file to run: 	random_walker_gui.py
Requires: 		functions.py, random_walker_3D.py, random_walker_4D.py, data in random_walker parent folder (ibt_4dFlow)

Workflow for best segmentation result:

1) Cycle through display modes ("Vel. Magnitude"-Button) to see the image with the velocity magnitude
2) Explore the 4D volume with the sliders "t-axis" and "z-axis" until you see good contrast in the velocity
3) Press "Scribble FG" to add a scribble to the foregound, draw a couple of lines inside the aorta
4) Press "Scribble BG" and repeat with the background
5) T and z-axis are frozen (on purpose)
6) Press "Add scribble" to unfreeze and add more scribbles in other slices
7) Add 2-3 more scribbles at other slices ("z-axis"-slider)
8) Press "Run 3D" to run the algorithm in 3D
9) Check the result with the "Overlap" button
10) Add more scribbles where desired (you can also remove scribbles bz drawing new ones on a slice where scribbles already exist, the latest scribble will be used)
10) To propagate the 3D result to a 4D segmentation press "3D -> Run 4D" (uses 3D segmentation as rough markers for other timesteps with dilation and erosion)
11) Alternatively one may directly run the algorithm in 4D (but this requires a lot of scribbles also across time), this will take some minutes to compute
12) Done!

To reset and rerun the RW simply press "Quit" and restart the program

---------------------------------------------------------

