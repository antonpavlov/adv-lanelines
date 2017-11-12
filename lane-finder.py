# TODO:Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# TODO:Apply a distortion correction to raw images.
# TODO:Use color transforms, gradients, etc., to create a thresholded binary image.
# TODO:Apply a perspective transform to rectify binary image ("birds-eye view").
# TODO:Detect lane pixels and fit to find the lane boundary.
# TODO:Determine the curvature of the lane and vehicle position with respect to center.
# TODO:Warp the detected lane boundaries back onto the original image.
# TODO:Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# Imports

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Calibration
# Object points

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')


nx = 9  # Counted on image
ny = 5  # Counted on image

# Setup object points
objp = np.zeros((ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # Real world space
imgpoints = []  # Image plane


for fname in images:
    # Open file
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret is True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        #plt.imshow(img)
        #plt.show()  # Interactive mode
        #plt.savefig(fname[:-4] + '_processed.png')

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Undistort image, if you want
        #undist = cv2.undistort(img, mtx, dist, None, mtx)
        #plt.imshow(undist)
        #plt.savefig(fname[:-4] + '_undistorted.png')
