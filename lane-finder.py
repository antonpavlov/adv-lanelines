# -*- coding: utf-8 -*-
# Advanced Lane Finding - Udacity Self-Driving Car Engineer Nanodegree - Fall 2018
# Plan:
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# Apply a distortion correction to raw images.
# Use color transforms, gradients, etc., to create a thresholded binary image.
# Apply a perspective transform to rectify binary image ("birds-eye view").
# Detect lane pixels and fit to find the lane boundary.
# Determine the curvature of the lane and vehicle position with respect to center.
# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# Imports
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip


def calibration():
    """
    Camera calibration function for compensation of radial and tangential distortions.
    :return: Success flag and correction coefficients
    Ref: Course notes and https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    """
    # ./camera_cal/calibration1.jpg nx=9 ny=5
    # ./camera_cal/calibration2.jpg nx=9 ny=6
    # ./camera_cal/calibration3.jpg nx=9 ny=6
    # ./camera_cal/calibration4.jpg nx=5 ny=6
    # ./camera_cal/calibration5.jpg nx=7 ny=6
    # ./camera_cal/calibration6.jpg nx=9 ny=6
    # ...
    # ./camera_cal/calibration20.jpg nx=9 ny=6

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

        # Skip following files; nx and ny should be 9 and 6 only
        if fname == './camera_cal/calibration1.jpg':
            #print("Break!")  # Debug
            pass
        elif fname == './camera_cal/calibration4.jpg':
            #print("Break!")  # Debug
            pass
        elif fname == './camera_cal/calibration5.jpg':
            #print("Break!")  # Debug
            pass
        else:
            # Open an input image
            image = cv2.imread(fname)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
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
    # End of FOR loop
    return cal_mtx, cal_dist


def select_yellow(image):
    """
    Threshold to select a yellow color marks
    :param image: Undistorted image
    :return: Thresholded image
    Ref: Udacity review
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([20, 60, 60])
    upper = np.array([38, 174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def select_white(image):
    """
    Threshold to select a white color marks
    :param image: Undistorted image
    :return: Thresholded image
    Ref: Udacity review
    """
    #lower = np.array([202, 202, 202])
    lower = np.array([190, 190, 190])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)
    return mask


def absolute_sobel_threshold(img, orient='x', thresh=(0, 255)):
    """
    Application of Sobel operator on undistorted image
    :param img: Undistorted image
    :param orient: Direction of gradient
    :param thresh: Default threshold tuple
    :return: Thresholded binary image
    Ref: Course notes
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    # is equal or more than thresh_min and less than thresh_max
    result = np.zeros_like(scaled_sobel)
    result[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return result


def magnitude_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Filter an image by gradient magnitude in both (x and y) directions
    :param img: Undistorted image
    :param sobel_kernel: A kernel size of Sobel transform
    :param mag_thresh: Default threshold tuple
    :return: Thresholded binary image
    Ref: Course notes
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
    result = np.zeros_like(gradmag)
    result[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return result


def direction_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Filter an image considering gradient orientation
    :param img: Undistorted image
    :param sobel_kernel: A kernel size of Sobel transform
    :param thresh: Default threshold tuple
    :return: Thresholded binary image
    Ref: Course notes
    """
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absolute_gradient_direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    result = np.zeros_like(absolute_gradient_direction)
    result[(absolute_gradient_direction >= thresh[0]) & (absolute_gradient_direction <= thresh[1])] = 1
    return result


def hlscolor_threshold(img, thresh=(0, 255)):
    """
    HLS color space threshold
    :param img: Undistorted image
    :param thresh: Default threshold values
    :return: Thresholded binary image
    Ref: Course notes
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    #result = np.zeros_like(s_channel)
    #result[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    yellow = select_yellow(img)
    white = select_white(img)

    result = np.zeros_like(yellow)
    result[(yellow >= 1) | (white >= 1)] = 1

    result[((s_channel > thresh[0]) | ((s_channel <= thresh[1])
                                                & (yellow == 1))) | white == 1] = 1


    return result


def all_combined_threshold(input_image):
    """
    Apply all thresholds to undistorted image
    :param input_image: Undistorted image
    :return: Combined binary image
    Ref: Course notes
    """
    # Sobel kernel size
    ksize = 3  # Should be an odd number to smooth a gradient

    # Apply threshold functions
    absolute_binary = absolute_sobel_threshold(input_image, orient='x', thresh=(50, 255))
    magnitude_binary = magnitude_threshold(input_image, sobel_kernel=ksize, mag_thresh=(50, 255))
    direction_binary = direction_threshold(input_image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    hlscolor_binary = hlscolor_threshold(input_image, thresh=(170, 255))
    yellow = select_yellow(input_image)
    white = select_white(input_image)

    # Combine threshold results in one binary image
    combine_all_binary = np.zeros_like(dir_binary)

    # Reviewer suggestion to consider colors
    #combine_all_binary[(absolute_binary == 1 | ((magnitude_binary == 1)
    #                 & (direction_binary == 1 | ((hlscolor_binary == 1)
    #                 & (yellow == 1))))) | white == 1] = 1

    combine_all_binary[(absolute_binary == 1 | ((magnitude_binary == 1)
                                                & (direction_binary == 1))) | hlscolor_binary == 1] = 1

    return combine_all_binary


def perspective_warp(img):
    """
    Execute perspective transform; warp in image
    :param img: Filtered image
    :return: Warped image and perspective transform results
    Ref: https://github.com/georgesung/advanced_lane_detection
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
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)

    return warped, m, m_inv


def find_lanes(binary_warped):
    """
    Find lanes in a binary warped image
    :param binary_warped: Undistorted and warped image
    :return: An image and data of lane location within the output image
    Ref: Course notes
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    ploty = np.linspace(0, warped_image.shape[0] - 1, warped_image.shape[0])
    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit


def find_lanes_secondary(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    ## Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    #left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    #left_line_pts = np.hstack((left_line_window1, left_line_window2))
    #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    #right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return out_img


def curvature_calc(lanes_left_fitx, lanes_right_fitx, ploty):
    """
    Calculation of lane curvature for both boundaries
    :param lanes_left_fitx:
    :param lanes_right_fitx:
    :param ploty: an image prototype
    :return: values of the left and right lane curvature in meters
    Ref: Course notes
    """
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, lanes_left_fitx, 2)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit = np.polyfit(ploty, lanes_right_fitx, 2)
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, lanes_left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, lanes_right_fitx * xm_per_pix, 2)

    # Calculate the new radius of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    return left_curverad, right_curverad


def draw_over(warped, undist, lanes_left_fitx, lanes_right_fitx, ploty, Minv):
    """
    Draw results of processing over original image
    :param warped: Warped image
    :param undist: Undistorted image
    :param lanes_left_fitx: Lane lines data
    :param lanes_right_fitx: Lane lines data
    :param ploty: An image prototype
    :param Minv: Inverse perspective transform
    :return: Resulting image
    Ref: Course notes
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([lanes_left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([lanes_right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, Minv, (1280, 720))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, new_warp, 0.3, 0)
    return result


def find_vehicle_offset(undist, left_fit, right_fit):
    """
    Calculate vehicle offset from lane center, in meters
    :param undist: undistorted image
    :param left_fit: Left lane boundary
    :param right_fit: Right lane boundary
    :return: value of vehicle offset within the lane in meters
    Ref: https://github.com/georgesung/advanced_lane_detection
    """
    # Calculate vehicle center offset in pixels
    bottom_y = undist.shape[0] - 1
    bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
    bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
    vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

    # Convert pixel offset to meters
    xm_per_pix = 3.7/700  # meters per pixel in x dimension
    vehicle_offset *= xm_per_pix
    return vehicle_offset


def process_image(image):
    """
    Video processing function
    :param image: frame image
    :return: processed image
    """
    undist_image = cv2.undistort(image, mtx, dist, None, mtx)
    combined_binary = all_combined_threshold(undist_image)
    warped_image, M, Minv = perspective_warp(combined_binary)
    lanes_img, lanes_left_fitx, lanes_right_fitx, ploty, left_fit, right_fit = find_lanes(warped_image)
    left_curve, right_curve = curvature_calc(lanes_left_fitx, lanes_right_fitx, ploty)
    curve_avg = (left_curve + right_curve) / 2
    upper_message = "Radius of curvature is " + str(round(curve_avg, 2)) + " meters"
    img_lanes = draw_over(warped_image, undist_image, lanes_left_fitx, lanes_right_fitx, ploty, Minv)
    offset = find_vehicle_offset(undist_image, left_fit, right_fit)
    if offset <= 0:
        lower_message = "Vehicle is " + str(abs(round(offset, 2))) + " meters left of center"
    else:
        lower_message = "Vehicle is " + str(abs(round(offset, 2))) + " meters right of center"
    final_image = cv2.putText(img_lanes, text=upper_message, org=(20, 70), fontFace=2, fontScale=1,
                              color=(0, 0, 255), thickness=4)
    final_image = cv2.putText(img_lanes, text=lower_message, org=(20, 130), fontFace=2, fontScale=1,
                              color=(0, 0, 255), thickness=4)
    return final_image

# Define global variables of calibration coefficients
global mtx, dist


if __name__ == "__main__":

    DEBUG = False  # Save all intermediate images

    # Calibrate camera
    mtx, dist = calibration()

    # Read input images from folder
    test_images = glob.glob('./test_images/*.jpg')

    for idx, f_name in enumerate(test_images):
        # Undistort the image
        input_image = cv2.imread(f_name)

        if DEBUG is True:
            ax1 = plt.clf()
            ax1 = plt.imshow(input_image)
            ax1 = plt.savefig(f_name[:-4] + '_A_processed.png')

        undist_image = cv2.undistort(input_image, mtx, dist, None, mtx)

        if DEBUG is True:
            ax2 = plt.clf()
            ax2 = plt.imshow(undist_image)
            ax2 = plt.savefig(f_name[:-4] + '_B_undistort.png')

        # Use color transforms, gradients, etc., to create a thresholded binary image.
        # Absolute horizontal Sobel operator on the image
        grad_binary = absolute_sobel_threshold(undist_image, orient='x', thresh=(50,255))

        if DEBUG is True:
            ax3 = plt.clf()
            ax3 = plt.imshow(grad_binary)
            ax3 = plt.savefig(f_name[:-4] + '_C_abs_sobel_thresh.png')

        # Sobel operator in both horizontal and vertical directions and calculate its magnitude
        mag_binary = magnitude_threshold(undist_image, sobel_kernel=3, mag_thresh=(50, 255))

        if DEBUG is True:
            ax4 = plt.clf()
            ax4 = plt.imshow(mag_binary)
            ax4 = plt.savefig(f_name[:-4] + '_D_mag_thresh.png')

        # Sobel operator to calculate the direction of the gradient
        dir_binary = direction_threshold(undist_image, sobel_kernel=15, thresh=(0.7, 1.3))

        if DEBUG is True:
            ax5 = plt.clf()
            ax5 = plt.imshow(dir_binary)
            ax5 = plt.savefig(f_name[:-4] + '_E_dir_binary.png')

        # Convert the image from RGB space to HLS space, and threshold the S channel
        hls_binary = hlscolor_threshold(undist_image, thresh=(170, 255))

        if DEBUG is True:
            ax6 = plt.clf()
            ax6 = plt.imshow(hls_binary)
            ax6 = plt.savefig(f_name[:-4] + '_F_hls_select.png')

        # Combine the above binary images to create the final binary image
        combined_binary = all_combined_threshold(undist_image)

        if DEBUG is True:
            ax7 = plt.clf()
            ax7 = plt.imshow(combined_binary)
            ax7 = plt.savefig(f_name[:-4] + '_G_combined_thresh.png')

        # Apply a perspective transform to rectify binary image ("birds-eye view")
        warped_image, M, Minv = perspective_warp(combined_binary)

        if DEBUG is True:
            ax8 = plt.clf()
            ax8 = plt.imshow(warped_image)
            ax8 = plt.savefig(f_name[:-4] + '_H_perspective.png')

        # Find lanes
        lanes_img, lanes_left_fitx, lanes_right_fitx, ploty, left_fit, right_fit = find_lanes(warped_image)

        if DEBUG is True:
            ax9 = plt.clf()
            ax9 = plt.imshow(lanes_img)
            ax9 = plt.plot(lanes_left_fitx, ploty, color='yellow')
            ax9 = plt.plot(lanes_right_fitx, ploty, color='yellow')
            ax9 = plt.xlim(0, 1280)
            ax9 = plt.ylim(720, 0)
            ax9 = plt.savefig(f_name[:-4] + '_I_lanes.png')

        # Curvature calculation
        left_curve, right_curve = curvature_calc(lanes_left_fitx, lanes_right_fitx, ploty)
        curve_avg = (left_curve + right_curve) / 2
        print("Filename: ", f_name, " Left:", left_curve, " Right:", right_curve, " Average:", curve_avg)
        upper_message = "Radius of curvature is " + str(round(curve_avg,2)) + " meters"

        # Draw lanes over undistorted image
        img_lanes = draw_over(warped_image, undist_image, lanes_left_fitx, lanes_right_fitx, ploty, Minv)

        if DEBUG is True:
            ax10 = plt.clf()
            ax10 = plt.imshow(img_lanes)
            ax10 = plt.savefig(f_name[:-4] + '_J_withLanes.png')

        # Vehicle offset
        offset = find_vehicle_offset(undist_image, left_fit, right_fit)
        print("Filename: ", f_name, " Vehicle offset: ", offset)
        if offset <= 0:
            lower_message = "Vehicle is " + str(abs(round(offset, 2))) + " meters left of center"
        else:
            lower_message = "Vehicle is " + str(abs(round(offset, 2))) + " meters right of center"

        # Overlay parameters
        final_image = cv2.putText(img_lanes, text=upper_message, org=(20, 70), fontFace=2, fontScale=1,
                                   color=(0, 0, 255), thickness=4)

        final_image = cv2.putText(img_lanes, text=lower_message, org=(20, 130), fontFace=2, fontScale=1,
                                  color=(0, 0, 255), thickness=4)

        # Save result
        ax11 = plt.clf()
        ax11 = plt.imshow(final_image)
        ax11 = plt.savefig(f_name[:-4] + '_K_final.png')
    # End of the FOR loop of a sequence of test images

    # Process video
    video_input = VideoFileClip("videos/project_video.mp4").subclip(35, 45)
    video_output = 'videos/OUTPUT_VIDEO.mp4'

    output_clip = video_input.fl_image(process_image)
    output_clip.write_videofile(video_output, audio=False)