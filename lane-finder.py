# -*- coding: utf-8 -*-
# Plan:
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# Apply a distortion correction to raw images.
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


def calibration():
    """
    Camera calibration function for compensation of radial and tangential distortions. [1]
    This function receives a calibration chess board image and number of inside corner in it. The return values are
    a flag of function execution status, camera matrix, and distortion coefficients;

    Args:
        none: Calibration function is developed for specific camera and specific calibration pictures. All input
              parameters like nx and ny are hardcoded. Provided images have the following nx/ny coefficients:
              ./camera_cal/calibration1.jpg nx=9 ny=5
              ./camera_cal/calibration2.jpg nx=9 ny=6
              ./camera_cal/calibration3.jpg nx=9 ny=6
              ./camera_cal/calibration4.jpg nx=5 ny=6
              ./camera_cal/calibration5.jpg nx=7 ny=6
              ./camera_cal/calibration6.jpg nx=9 ny=6
              ...
              ./camera_cal/calibration20.jpg nx=9 ny=6
              Calibration function won't work if nx and ny are incorrect. There is no homogeneity among nx and ny on
              the first 5 images. That's why we are going to use images calibration6.jpg to calibration20.jpg.

    Returns:
        success_flag: True for success, False otherwise.
        cal_mtx: Camera matrix
        cal_dist:  Distortion coefficients

    [1]: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    """
    # Setup object points
    nx = 9  # Internal corners on horizontal
    ny = 6  # Internal corners on vertical
    object_point = np.zeros((ny * nx, 3), np.float32)
    object_point[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points
    object_points = []  # Real world space
    image_points = []   # Image plane

    # Image List
    images = glob.glob('./camera_cal/calibration*.jpg')

    for idx, fname in enumerate(images):
        # Open an input image
        input_image = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        success_flag = False  # Let's assume that there is no corners and function below fails
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret is True:
            success_flag = True
            image_points.append(corners)
            object_points.append(object_point)

            # Calibrate camera
            cal_ret, cal_mtx, cal_dist, cal_rvecs, cal_tvecs = cv2.calibrateCamera(object_points,
                                                                               image_points,
                                                                               gray.shape[::-1], None, None)
        else:
            cal_mtx = 0
            cal_dist = 0

    return success_flag, cal_mtx, cal_dist


if __name__ == "__main__":

    # Calibrate camera
    success_flag, mtx, dist = calibration()
    if success_flag is False:
        raise Exception('Calibration function failed!')

    # Read input images from folder
    test_images = glob.glob('./test_images/*.jpg')

    for idx, fname in enumerate(test_images):
        # Undistort the image
        input_image = cv2.imread(fname)
        undist_image = cv2.undistort(input_image, mtx, dist, None, mtx)

        # Tests if undistort works
        #ax1 = plt.imshow(input_image)
        #ax1 = plt.savefig(fname[:-4] + '_processed.png')
        #ax2 = plt.imshow(undist_image)
        #ax2 = plt.savefig(fname[:-4] + '_undistorted.png')
