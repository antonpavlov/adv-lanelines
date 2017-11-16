# -*- coding: utf-8 -*-
# Plan:
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# Apply a distortion correction to raw images.
# Use color transforms, gradients, etc., to create a thresholded binary image.
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

from moviepy.editor import VideoFileClip


def calibration():
    # TODO: Ignore calibration1.jpg to 6 files.
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
        image = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        success_flag = False  # Let's assume that there is no corners and function below fails
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        cal_mtx = []
        cal_dist = []
        # If found, draw corners
        if ret is True:
            success_flag = True
            image_points.append(corners)
            object_points.append(object_point)

            # Calibrate camera
            cal_ret, cal_mtx, cal_dist, cal_rvecs, cal_tvecs = cv2.calibrateCamera(object_points,
                                                                               image_points,
                                                                               gray.shape[::-1], None, None)
    return success_flag, cal_mtx, cal_dist


def absolute_sobel_threshold(img, orient='x', thresh_min=0, thresh_max=255):
    """
    Description pending

    Args:
        none:

    Returns:
        none:
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Just assume default 'x', correct after. Yeap, it's not so elegant...

    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    else:
        raise Exception("Wrong orientation parameter!")

    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output


def magnitude_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Description pending

    Args:
        none:

    Returns:
        none:
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def direction_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Description pending

    Args:
        none:

    Returns:
        none:
    """
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hlscolor_threshold(img, thresh=(0, 255)):
    """
    Description pending

    Args:
        none:

    Returns:
        none:
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def all_combined_threshold(input_image):
    """
    Description pending

    Args:
        none:

    Returns:
        none:
    """
    absolute_binary = absolute_sobel_threshold(input_image, orient='x', thresh_min=50, thresh_max=255)
    magnitude_binary = magnitude_threshold(input_image, sobel_kernel=3, mag_thresh=(50, 255))
    direction_binary = direction_threshold(input_image, sobel_kernel=15, thresh=(0.7, 1.3))
    hlscolor_binary = hlscolor_threshold(input_image, thresh=(170, 255))
    combine_all_binary = np.zeros_like(dir_binary)
    combine_all_binary[(absolute_binary == 1 | ((magnitude_binary == 1)
                                                & (direction_binary == 1))) | hlscolor_binary == 1] = 1
    return combine_all_binary


def perspective_transform(img):
    """
    Execute perspective transform
    """
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
         [[200, 720],
         [1100, 720],
         [595, 450],
         [685, 450]])
    dst = np.float32(
         [[300, 720],
         [980, 720],
         [300, 0],
         [980, 0]])

    m = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)

    return warped


def process_image(image):
    undist_image = cv2.undistort(image, mtx, dist, None, mtx)
    #ax = plt.imshow(undist_image)
    #ax = plt.savefig('undist_image.png')
    combined_binary = all_combined_threshold(undist_image)
    #bx = plt.imshow(combined_binary)
    #bx = plt.savefig('combined_binary.png')
    return combined_binary

# Define global variables of calibration coefficients
global mtx, dist


if __name__ == "__main__":

    DEBUG = False  # Save all intermediate images

    # Calibrate camera
    s_flag, mtx, dist = calibration()
    if s_flag is False:
        raise Exception('Calibration function failed!')

    # Read input images from folder
    test_images = glob.glob('./test_images/*.jpg')

    for idx, f_name in enumerate(test_images):
        # Undistort the image
        input_image = cv2.imread(f_name)

        if DEBUG is True:
            # Save input file for comparison purposes after
            ax1 = plt.imshow(input_image)
            ax1 = plt.savefig(f_name[:-4] + '_A_processed.png')

        undist_image = cv2.undistort(input_image, mtx, dist, None, mtx)

        if DEBUG is True:
            # Tests of undistort function
            ax2 = plt.imshow(undist_image)
            ax2 = plt.savefig(f_name[:-4] + '_B_undistort.png')

        # Use color transforms, gradients, etc., to create a thresholded binary image.
        # Absolute horizontal Sobel operator on the image
        grad_binary = absolute_sobel_threshold(undist_image, orient='x', thresh_min=20, thresh_max=100)

        if DEBUG is True:
            # Tests of abs_sobel_thresh function
            ax3 = plt.imshow(grad_binary)
            ax3 = plt.savefig(f_name[:-4] + '_C_abs_sobel_thresh.png')

        # Sobel operator in both horizontal and vertical directions and calculate its magnitude
        mag_binary = magnitude_threshold(undist_image, sobel_kernel=3, mag_thresh=(30, 100))

        if DEBUG is True:
            # Tests of mag_thresh function
            ax4 = plt.imshow(mag_binary)
            ax4 = plt.savefig(f_name[:-4] + '_D_mag_thresh.png')

        # Sobel operator to calculate the direction of the gradient
        dir_binary = direction_threshold(undist_image, sobel_kernel=15, thresh=(0.7, 1.3))

        if DEBUG is True:
            # Tests of dir_threshold function
            ax5 = plt.imshow(dir_binary)
            ax5 = plt.savefig(f_name[:-4] + '_E_dir_binary.png')

        # Convert the image from RGB space to HLS space, and threshold the S channel
        hls_binary = hlscolor_threshold(undist_image, thresh=(90, 255))

        if DEBUG is True:
            # Tests of hls_select function
            ax6 = plt.imshow(hls_binary)
            ax6 = plt.savefig(f_name[:-4] + '_F_hls_select.png')

        # Combine the above binary images to create the final binary image
        combined_binary = all_combined_threshold(undist_image)

        if DEBUG is True:
            # Tests of hls_select function
            ax7 = plt.imshow(combined_binary)
            ax7 = plt.savefig(f_name[:-4] + '_G_combined_thresh.png')

        # Apply a perspective transform to rectify binary image ("birds-eye view")
        warped_image = perspective_transform(combined_binary)

        if DEBUG is True:
            # Tests of hls_select function
            ax8 = plt.imshow(warped_image)
            ax8 = plt.savefig(f_name[:-4] + '_H_perspective.png')
    # End of the FOR loop of a sequence of test images

    # Process video
    video_input = VideoFileClip("videos/project_video.mp4").subclip(0, 5)
    video_output = 'videos/OUTPUT_VIDEO.mp4'

    output_clip = video_input.fl_image(process_image)
    output_clip.write_videofile(video_output, audio=False)


