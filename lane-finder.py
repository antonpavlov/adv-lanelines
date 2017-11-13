# -*- coding: utf-8 -*-
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
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
import matplotlib.pyplot as plt


def calibration(input_image, nx, ny):
    """Camera calibration function for compensation of radial and tangential distortions. [1]
        This function receives a calibration chess board image and number of inside corner in it. The return values are
        a flag of function execution status, camera matrix, and distortion coefficients;

        Args:
            input_image (np.ndarray, uint8): The input image.
            nx (int): Number of inside corners in x.
            ny (int): Number of inside corners in y.

        Returns:
            success_flag: True for success, False otherwise.
            cal_mtx: Camera matrix
            cal_dist:  Distortion coefficients
            
        [1]: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    """
    # Setup object points
    object_point = np.zeros((ny * nx, 3), np.float32)
    object_point[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points
    object_points = []  # Real world space
    image_points = []   # Image plane

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
    # ./camera_cal/calibration1.jpg nx=9 ny=5
    # ./camera_cal/calibration2.jpg nx=9 ny=6
    # ./camera_cal/calibration3.jpg nx=9 ny=6
    # ./camera_cal/calibration4.jpg nx=5 ny=6
    # ./camera_cal/calibration5.jpg nx=7 ny=6
    # ./camera_cal/calibration6.jpg nx=9 ny=6
    # ./camera_cal/calibration7.jpg nx= ny=
    # ./camera_cal/calibration8.jpg nx= ny=
    # ./camera_cal/calibration9.jpg nx= ny=
    # ./camera_cal/calibration10.jpg nx= ny=
    # ./camera_cal/calibration11.jpg nx= ny=
    # ./camera_cal/calibration12.jpg nx= ny=
    # ./camera_cal/calibration13.jpg nx= ny=
    # ./camera_cal/calibration14.jpg nx= ny=
    # ./camera_cal/calibration15.jpg nx= ny=
    # ./camera_cal/calibration16.jpg nx= ny=
    # ./camera_cal/calibration17.jpg nx= ny=
    # ./camera_cal/calibration18.jpg nx= ny=
    # ./camera_cal/calibration19.jpg nx= ny=
    # ./camera_cal/calibration20.jpg nx= ny=
    image_filename = './camera_cal/calibration8.jpg'
    image = cv2.imread(image_filename)
    ax1 = plt.imshow(image)
    ax1 = plt.savefig(image_filename[:-4] + '_processed.png')
    # Calibrate camera
    success_flag, mtx, dist = calibration(image, nx=9, ny=6)
    if success_flag is False:
        print("Calibration function failed!")
        exit(0)
    # Undistort the image
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    ax2 = plt.imshow(undist)
    ax2 = plt.savefig(image_filename[:-4] + '_undistorted.png')



# Calibration
# Object points

# Make a list of calibration images
#images = glob.glob('./camera_cal/calibration*.jpg')


#nx = 9  # Counted on image
#ny = 5  # Counted on image

# Setup object points
#objp = np.zeros((ny*nx, 3), np.float32)
#objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points
#objpoints = []  # Real world space
#imgpoints = []  # Image plane


#for fname in images:
    # Open file
#    img = cv2.imread(fname)

    # Convert to grayscale
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
#    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
#    if ret is True:
#        imgpoints.append(corners)
#        objpoints.append(objp)

        # Draw and display the corners
#        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
#        plt.imshow(img)
#        #plt.show()  # Interactive mode
#        plt.savefig(fname[:-4] + '_processed.png')

        # Calibrate camera
#        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Undistort image, if you want
#        undist = cv2.undistort(img, mtx, dist, None, mtx)
        #plt.imshow(undist)
#        plt.savefig(fname[:-4] + '_undistorted.png')
