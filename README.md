# adv-lanelines
Advanced Lane Finding - Udacity Self-Driving Car Engineer Nanodegree. 

The goal of this project is to build an *advanced* software pipeline for an automatic recognition of a road surface markings. A *simpler* version of this project can be found here: [https://github.com/antonpavlov/lanelines](https://github.com/antonpavlov/lanelines). This repository strongly relies and contains references to Udacity's self-driving car nanodegree program.

### Contents of the repo ###
`<camera_cal>` - A folder with calibration images <br />
`<support_files>` - A folder with images used in README <br />
`<test_images>` - A folder with test images; Results of processing will be saved there. <br /> 
`<videos>` - A folder with the test video; Result will be saved there. <br />
`.gitignore` - .gitignore for Python. <br />
`LICENSE` - The MIT license text. <br />
`README.md` - this file. <br />
`lane-finder.py` - The script for lane finding.

### Environment set-up ###

Before run a script from this repo, please setup an environment with all required dependencies: [https://github.com/udacity/CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
    
Download also an input data from Udacity's project repository: [https://github.com/udacity/CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines)

Clone this repository somewhere and copy from there the `lane-finder.py` script only into the **CarND-Advanced-Lane-Lines** folder. I suggest to use [Anaconda](https://www.anaconda.com/download/).

### Reflection ###
The following approach was suggested during the course:
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Technical restrictions and weaknesses ###
- Calibration coefficients are related to a specific camera used to record images. 
- Script in this repo works only with 1280 X 720 images.
- The proposed pipeline has several weaknesses related to lane marks recognition. Some of them are related to lightning conditions, time of the day, and weather (rain and snow); some of them are related to road surface quality.
- The pipeline is not able to do any inference regarding lane marks and vehicle position when part of the lane ahead is obstructed by another car.
- The pipeline relies on existence of the road marking on that particular road under evaluation. 
- Unfortunately, most of coefficients of a transforms and a thresholds are specified on empirical basis considering only images, proposed by Udacity. 


### Examples of processing ###
Camera calibration - Original image
![Original](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/calibration1_processed.png)


Camera calibration - Undistorted image
![Undistorted](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/calibration1_undistorted.png)

<br />

Let's build the following pipeline:

1. Open an image

![Processed](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_A_processed.png)

<br />


2. Apply image correction

![Undistort](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_B_undistort.png)

<br />


3. Application of Sobel operator on undistorted image 

![Gradient](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_C_abs_sobel_thresh.png)

<br />


4. Filter an image by gradient magnitude in both (x and y) directions 

![Magnitude](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_D_mag_thresh.png)

<br />


5. Filter an image considering gradient orientation 

![Orientation](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_E_dir_binary.png)

<br />


6. HLS color space threshold 

![HLS](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_F_hls_select.png)

<br />


7. All thresholds applied together to undistorted image

![All_together](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_G_combined_thresh.png)

<br />


8. Perspective transform; warp-in image

![Perspective](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_H_perspective.png)

<br />


9. Find lanes in a binary warped image

![Lanes](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_I_lanes.png)

<br />


10. Make curvature calculations; vehicle position and draw results over an original image

![Lanes](https://github.com/antonpavlov/adv-lanelines/blob/master/support_files/test3_K_final.png)

<br />


### Future work ###
As a future work, it might be the case for an application of a newly proposed Hinton’s Capsule Networks. These networks may be trained to make an inference over lane existence and its bounds taking into account important spatial hierarchies of lane markings, other vehicles and road equipment. 


### License ###

Python script `lane-finder.py` is distributed under the terms described in the MIT license. 
Please refer to [Udacity](https://github.com/udacity) regarding all other supporting materials.