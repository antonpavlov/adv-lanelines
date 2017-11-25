# adv-lanelines
Advanced Lane Finding - Udacity Self-Driving Car Engineer Nanodegree. 

The goal of this project is to build an *advanced* software pipeline for an automatic recognition of road surface markings. A *simpler* version of this project can be found here: [https://github.com/antonpavlov/lanelines](https://github.com/antonpavlov/lanelines)

### Contents of the repo ###
<placeholder>

### Environment set-up ###

Before run a script this repo, please install an environment with all required dependencies: [https://github.com/udacity/CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
    
Download also an input data from Udacity's project repository: [https://github.com/udacity/CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines)

Clone this repository somewhere and copy from there the `lane-finder.py` script into the **CarND-Advanced-Lane-Lines** folder.

### Reflection ###
Plan:
- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- TODO:Warp the detected lane boundaries back onto the original image.
- TODO:Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Example of processing ###
<placeholder>

### License ###

Python script `lane-finder.py` is distributed under the terms described in the MIT license. 
Please refer to [Udacity](https://github.com/udacity) regarding all other supporting materials.