import cv2
import glob
import pickle
import numpy as np
from scipy import ndimage
import random
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

		
####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part A: Help Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
### plot_dist_vs_undist: plot an image before and after we calibrated the camera
###   Input: 
###		img_dist: original image in RGB.
###		img_undist: image after calibration in RGB.
###		corners: Indication if we are dealing with the Chessboard. If so this variable will hold the corners of the Chessboard. 
###     title: string for the title
###
###   Output: 
###      save the figure.
def plot_dist_vs_undist(img_dist, img_undist, corners=None, title=''):
	
	if corners is not None: # Chessboard case
		fig_name = 'chessboard'
		# crop the image show to the relevant section - only of the chessboard
		X_extra_from_corner = int(50+(max(corners[:,0,0]) - min(corners[:,0,0]))/8)
		Y_extra_from_corner = int(50+(max(corners[:,0,1]) - min(corners[:,0,1]))/5)
		X_min = int(max(0,-X_extra_from_corner+min(corners[:,0,0])))
		X_max = int(min(img_dist.shape[1],X_extra_from_corner+max(corners[:,0,0])))
		Y_min = int(max(0,-Y_extra_from_corner+min(corners[:,0,1])))
		Y_max = int(min(img_dist.shape[0],Y_extra_from_corner+max(corners[:,0,1])))
	else: # road case
		fig_name = 'road'
	
	# plot the two images - the Original and the Undistorted
	plt.figure(figsize=(16,8))
	plt.subplot(1,2,1)
	plt.imshow(img_dist)
	plt.title('Original: {}'.format(title))
	if corners is not None:
		plt.xlim([X_min,X_max])
		plt.ylim([Y_max,Y_min])
	plt.subplot(1,2,2)
	plt.imshow(img_undist)
	plt.title('Undistorted Image')
	if corners is not None:
		plt.xlim([X_min,X_max])
		plt.ylim([Y_max,Y_min])
	
	plt.savefig('output_images/original_vs_calibratted_{}.png'.format(fig_name))

	
### get_object_grid: return a grid given the max value in x,y axis. z axis is set to zero
###   Input: 
###		nx: number of grid points in the x-axis.
###		ny: number of grid points in the y-axis.
###
###   Output: 
###      objp: 2D array like: (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0).
def get_object_grid(nx, ny):
	objp = np.zeros((nx*ny,3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)      # x,y coordinates. z is set to zero.	
	return objp


### find_calibration_params: Find the calibration parameters, using the given calibration images
###   Note: not all the images that were given to us were 9x6 images (3/20). We made some changes to use also the other images
###   Input: 
###		nx: nominal number of corners in the x-axis. we will also check up to -dn around it
###		ny: nominal number of corners in the y-axis. we will also check up to -dn around it
###		dn: the maximal delta we will ad to each axis 
###     plot_en: after we finish with the calibration we choose a random calibration image and we plot the original image and the image after we calibrated it.
###
###   Output: 
###      image: the final image. In a RGB format.
def find_calibration_params(nx, ny, dn, plot_en=False):
	print('---> Start Calibration')
	# Read all the calibration images
	imgs_fnames = glob.glob('camera_cal/calibration*.jpg')
	
	# Arrays to store object points and image points from all the images.
	obj_points = [] # 3d points in real world space
	img_points = [] # 2d points in image plane.
	
	dn_vec = np.mgrid[-dn:1, -dn:1].T.reshape(-1,2)          # we will also check around nx,ny.
	dn_vec = np.flipud(dn_vec)                               # we will start with the original nx,ny
	for fname in imgs_fnames:
		img = cv2.imread(fname)                              # read each image (the image will be in BGR)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # convert image to gray
		for d in dn_vec:
			ret, corners = cv2.findChessboardCorners(gray, (nx+d[0],ny+d[1]), None)  # find the corners in the given image
			if ret==True:                                  # found corners, so lets append them to the obj_points
				objp = get_object_grid(nx+d[0],ny+d[1])    # objp is NOT the same for all pictures
				obj_points.append(objp)                    # append objp to the total list
				img_points.append(corners)                 # corners we found in this image
				break

	print('\t-->Calibrate on {} images out of {} images'.format(len(img_points), len(imgs_fnames)))
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
	
	# Choose a random calibration image and we plot the original image and the image after we calibrated it 
	if plot_en:
		idx = np.random.randint(low=0, high=len(imgs_fnames))      # choose a random calibration image 
		img = cv2.imread(imgs_fnames[idx])                         # read the image
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                  # Switch to RGB format
		dst = cv2.undistort(img, mtx, dist, None, mtx)             # undistorted image

		plot_dist_vs_undist(img, dst, corners=img_points[idx], title=imgs_fnames[idx])
	return ret, mtx, dist, rvecs, tvecs


### calibrate_road_image: calibrate and save a random example for a road image using the calibration parameters from the chessboard calibration
###   Input: 
###		img: input (distorted) image. if None we will read it from the test_images folder
###		mtx: the camera matrix
###		dist: distortion coefficients
###		idx: the index of the image we want from the test images (0-5) or straight_lines images (0-1). None - for random index.
###		fname: the string at the start of the files name
###		plot_en: if we want to plot the original image and the calibrated image
###
###   Output: 
###      dst: the undistorted image.	
###      save the image before and after calibration.	
def calibrate_road_image(img=None, mtx=None, dist=None, idx=None, fname='test', plot_en=False):
	if (img is None):
		imgs_fnames = glob.glob('test_images/'+fname+'*.jpg')
		if idx is None:
			idx = np.random.randint(low=0, high=len(imgs_fnames))  # choose a random road image
		idx = max(0,min(idx,len(imgs_fnames)))                     # making sure we are in the range
		img = cv2.imread(imgs_fnames[idx])                         # read the image
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                  # Switch to RGB format
		img_title = imgs_fnames[idx]
	else:
		img_title = fname
	dst = cv2.undistort(img, mtx, dist, None, mtx)             # undistorted image
	if plot_en:
		plot_dist_vs_undist(img, dst, corners=None, title=img_title)
	return dst
	

### sobel_binary_th: apply sobel on the gray image on x-axis and y-axis. 
###                     Then we use 4 different thresholds on |sobel_x|,|sobel_y|,|sobel|,abg(sobel) to get a binary picture with the lane lines
###                     out = (|sobel_x|&|sobel_y|) | (|sobel|&abg(sobel))
###   Input: 
###		rgb_img: rgb image
###		kernel_size: kernel size of the sobel. default is 3
###		plot_en: plot the grey sacle image, the final result in binary image and a colored image of each of the components (sobel_x, |sobel|, ang_sobel) 
###
###   Output: 
###      b_sobel_total: binary image after all the thresholds were applied.	
def sobel_binary_th(rgb_img, kernel_size=3, plot_en=False):
	#print('\t--> Start Sobel Binary Threshold')
	# 1) convert to gray scale
	gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)               # convert to gray scale

	# 2) Now we calculate all the sobels and sobel functions we need
	sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size) # sobel on x-axis
	sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size) # sobel on y-axis
	abs_sobel_x = np.absolute(sobel_x)                             # |sobel_x|
	abs_sobel_x = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))    # scale |sobel_x|
	abs_sobel_y = np.absolute(sobel_y)                             # |sobel_y|
	abs_sobel_y = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y))    # scale |sobel_y|
	mag_sobel = np.sqrt(sobel_x**2 + sobel_y**2)                   # |sobel| = sqrt(sobel_x^2 + sobel_y^2) 
	mag_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))          # scale |sobel|
	# for the angle we use a larger kernel_size to reduce noise
	sobel_x_lpf = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=(5*kernel_size))      # sobel on x-axis with bigger kernel
	sobel_y_lpf = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=(5*kernel_size))      # sobel on y-axis with bigger kernel
	ang_sobel = np.arctan2(np.absolute(sobel_y_lpf), np.absolute(sobel_x_lpf))      # abg(sobel) [-pi,pi)
	
	# we put in a list all the sobels we will use
	all_sobel_name=['abs_sobel_x','abs_sobel_y','mag_sobel','ang_sobel']
	all_sobel=[abs_sobel_x,abs_sobel_y,mag_sobel,ang_sobel] 
	
	# 3) creating binary images - each will threshold a different sobel information 
	#    We start by setting the thresholds
	all_th_en=['th', 'all_ones', 'all_zeros', 'all_zeros'] # threshold enablers: "all zeros", "all ones" or to use the threshols  
	all_th = np.array([[30,200], [80,100], [70,100], [0.7, 1.3]]) # the threshold for each sobel function
	
	b_all_sobel = [] # a list of the binary sobels after we use the threshold
	for i in range(len(all_sobel)):
		if ('all_ones' == all_th_en[i]):                            # all ones
			b_all_sobel.append(np.ones_like(all_sobel[i])) 
			#print('\t\t{}: all Ones'.format(all_sobel_name[i]))
		elif ('all_zeros' == all_th_en[i]):                         # all zeros
			b_all_sobel.append(np.zeros_like(all_sobel[i])) 
			#print('\t\t{}: all Zeros'.format(all_sobel_name[i]))
		else:                                                       # normal threshold
			b_sobel = np.zeros_like(all_sobel[i]) 
			b_sobel[(all_sobel[i] >= all_th[i,0]) & (all_sobel[i] <= all_th[i,1])] = 1    # apply threshold
			b_all_sobel.append(b_sobel) 
	
	
	# 4) combine all the binary images
	b_sobel_ax = np.zeros_like(b_all_sobel[0])
	b_sobel_ax[(b_all_sobel[0] == 1) & (b_all_sobel[1] == 1)] = 1
	b_sobel_mag_ang = np.zeros_like(b_all_sobel[2])
	b_sobel_mag_ang[(b_all_sobel[2] == 1) & (b_all_sobel[3] == 1)] = 1
	b_sobel_total = np.zeros_like(b_sobel_ax)
	b_sobel_total[(b_sobel_ax == 1) | (b_sobel_mag_ang == 1)] = 1
	
	if plot_en: # plot the binary image and the colored binary components
		color_binary = np.dstack((b_sobel_ax, np.zeros_like(b_sobel_ax), b_sobel_mag_ang)) 
		plt.figure(figsize=(16,8))
		plt.subplot(1,3,1)
		plt.imshow(gray_img, cmap='gray')
		plt.title('Gray image')
		plt.subplot(1,3,2)
		plt.imshow(255*color_binary)
		plt.title('Sobel breakdown (r=sobel_ax, b=mag_ang(sobel))')
		plt.subplot(1,3,3)
		plt.imshow(b_sobel_total, cmap='gray')
		plt.title('Binary after sobel th')


		plt.savefig('output_images/applay_sobel_th.png')
	
	return b_sobel_total

	
### color_binary_th: apply color threshold on the different channels in the RGB, HLS color space 
###                    out = (r_channel | s_channel)
###   Input: 
###		rgb_img: rgb image
###		plot_en: plot the undistorted image, the final result in binary image and the colored binary of those two threshold. 
###
###   Output: 
###      b_color_total: binary image after all the thresholds were applied.	
def color_binary_th(rgb_img, plot_en=False):
	#print('\t--> Start Color Binary Threshold')
	# 1) take the relevant channels: red, Saturation
	r_img = rgb_img[:,:,0]                                       # Red channel 
	s_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)[:,:,2]      # Saturation channel 
	# we put in a list all the channels we will use
	all_clr_name=['red','sat']
	all_clr=[r_img,s_img] 
	
	# 2) creating binary images - each will threshold a different channel 
	#    We start by setting the thresholds
	all_th_en=['all_zeros', 'th'] # threshold enablers: "all zeros", "all ones" or to use the threshols  
	all_th = np.array([[170,255], [175,230]]) # the threshold for each channel
	
	b_all_clr = [] # a list of the binary channels after we use the threshold
	for i in range(len(all_clr)):
		if ('all_ones' == all_th_en[i]):                            # all ones
			b_all_clr.append(np.ones_like(all_clr[i])) 
			#print('\t\t{}: all Ones'.format(all_clr_name[i]))
		elif ('all_zeros' == all_th_en[i]):                         # all zeros
			b_all_clr.append(np.zeros_like(all_clr[i])) 
			#print('\t\t{}: all Zeros'.format(all_clr_name[i]))
		else:                                                       # normal threshold
			b_clr = np.zeros_like(all_clr[i]) 
			b_clr[(all_clr[i] >= all_th[i,0]) & (all_clr[i] <= all_th[i,1])] = 1    # apply threshold
			b_all_clr.append(b_clr) 
	

	# 4) combine all the binary images
	b_clr_total = np.zeros_like(b_all_clr[0])
	b_clr_total[(b_all_clr[0] == 1) | (b_all_clr[1] == 1)] = 1
	
	if plot_en: # plot the binary image and the colored binary components
		color_binary = np.dstack((b_all_clr[0], np.zeros_like(b_all_clr[0]), b_all_clr[1])) 
		plt.figure(figsize=(16,8))
		plt.subplot(1,3,1)
		plt.imshow(rgb_img)
		plt.title('Original image')
		plt.subplot(1,3,2)
		plt.imshow(255*color_binary)
		plt.title('Channel breakdown (r=red, b=sat)')
		plt.subplot(1,3,3)
		plt.imshow(b_clr_total, cmap='gray')
		plt.title('Binary after color th')

		plt.savefig('output_images/applay_color_th.png') 
	
	return b_clr_total


### apply_binary_th: apply the following thresholds:
###                    - sobel (all kind of functions)
###                    - color channel threshold in different color space 
###                        out = (sobel | color)
###   Input: 
###		rgb_img: rgb image
###		plot_en: plot the undistorted image, the final result in binary image and the colored binary of those two threshold. 
###
###   Output: 
###      b_total: binary image after all the thresholds were applied	
def apply_binary_th(rgb_img, plot_en=False):
	#print('---> Start Binary Threshold')
	
	b_sobel_undist_img = sobel_binary_th(rgb_img, kernel_size=3, plot_en=plot_en) # b_ stands for binary  
	b_color_undist_img = color_binary_th(rgb_img, plot_en=plot_en) # b_ stands for binary

	# combine all the binary images
	b_total = np.zeros_like(b_sobel_undist_img)
	b_total[(b_sobel_undist_img == 1) | (b_color_undist_img == 1)] = 1
	

	if plot_en: # plot the binary image and the colored binary components
		color_binary = np.dstack((b_sobel_undist_img, np.zeros_like(b_total),b_color_undist_img)) 
		plt.figure(figsize=(16,8))
		plt.subplot(1,3,1)
		plt.imshow(rgb_img)
		plt.title('Original image')
		plt.subplot(1,3,2)
		plt.imshow(255*color_binary)
		plt.title('Channel breakdown (r=sobel, b=color)')
		plt.subplot(1,3,3)
		plt.imshow(b_total, cmap='gray')
		plt.title('Binary after sobel&color th')

		plt.savefig('output_images/binary_final.png') 
	
	return b_total

	
### find_birdeye_matrix: Calc the bird-eye Tarnsform Matrix (and the inverse) 
###                      We will look on both images we have "straight_lines" and average the output
###   Input: 
###		M_op: wich matrix we want to return: 0-average on both straight_lines images. 1:straight_lines1, 2: straight_lines2
###		plot_en: plot the original image and the bird-eye image (for both pictures we have)
###			mtx: calibration matrix. if we want to plot the straight_lines images we will need the calibration matrix
###			dist: distortion coefficients. if we want to plot the straight_lines images we will need the distortion coefficients
###
###   Output: 
###      M: The tarnsform Matrix (normal to bird-eye)	
###      M_inv: The inverse tarnsform Matrix (bird-eye to normal)	
def find_birdeye_matrix(M_op=0, plot_en=False, mtx=None, dist=None):
	print('---> Start Calc the bird-eye Tarnsform Matrix')
	
	# 1) How many images do we have and the size of them
	prefix='straight_lines'                                # the prefix for the 'straight_lines' images
	imgs_fnames = glob.glob('test_images/'+prefix+'*.jpg') # all the images with straight_lines
	n_img = len(imgs_fnames)                               # number of images
	rgb_undist_img = calibrate_road_image(None, mtx, dist, idx=0, fname=prefix, plot_en=False) # load a image 
	img_size = (rgb_undist_img.shape[1], rgb_undist_img.shape[0])  # image size
	if False: # just to find the points on the original image
		plt.figure(figsize=(16,8))
		plt.imshow(rgb_undist_img)
		plt.show()
	
	# 2) we set the points for the trapeze of source images and for the bird-eye image
	#    Each trapeze is defined by 6 values: y_down, y_up, x_right_down, x_left_down, x_right_down, x_left_up
	y_down = [689,689]           # for the two images
	y_up = [450,450]             # for the two images
	x_right_down = [1055,1062]   # for the two images
	x_left_down = [250,260]      # for the two images
	x_right_up = [683,688]       # for the two images
	x_left_up = [596,595]        # for the two images
		
	# 3) Using getPerspectiveTransform we will calculate the transform matrix (and the inverse)  per image
	src=[]    # will hold all the sources points. trapeze (4 points) for each image
	dst=[]    # will hold all the destination points. rectangle (4 points) for each image
	M=[]      # will hold the transform matrix per image
	M_inv=[]  # will hold the inverse transform matrix per image
	for i in range(n_img):
		src.append(np.float32([[x_left_down[i],y_down[i]], [x_left_up[i]  ,y_up[i]],     [x_right_up[i]  ,y_up[i]],     [x_right_down[i],y_down[i]]]))
		dst.append(np.float32([[x_left_down[i],y_down[i]], [x_left_down[i],0], [x_right_down[i],0], [x_right_down[i],y_down[i]]]))
		M.append(cv2.getPerspectiveTransform(src[-1], dst[-1]))
		M_inv.append(cv2.getPerspectiveTransform(dst[-1], src[-1]))
	
	# 4) Average the transform matrix (and the inverse)
	M_avg = np.average(np.array(M),0)
	M_inv_avg = np.average(np.array(M_inv),0)
	
	if plot_en: # plot the straight_lines (after removing distortion) and their birdeye view
		plt.figure(figsize=(16,8))
		for i in range(n_img):
			rgb_undist_img = calibrate_road_image(None, mtx, dist, idx=i, fname=prefix, plot_en=False)              # get the calibrated image
			cv2.line(rgb_undist_img, (src[i][0,0],src[i][0,1]), (src[i][1,0],src[i][1,1]), [255,0,0], 2)      # draw the lane lines in the car view image - left lane
			cv2.line(rgb_undist_img, (src[i][2,0],src[i][2,1]), (src[i][3,0],src[i][3,1]), [255,0,0], 2)      # draw the lane lines in the car view image - right lane
			plt.subplot(2,2,i+1)
			plt.imshow(rgb_undist_img)
			plt.title('Car view:{}'.format(imgs_fnames[i]))
			
			warped = cv2.warpPerspective(rgb_undist_img, M[i], img_size)                                      # apply the transformation
			plt.subplot(2,2,i+3)
			plt.imshow(warped)
			plt.title('Birdeye view:{}'.format(imgs_fnames[i]))

		plt.savefig('output_images/birdeye_on_straight_lines.png')
	
	# 5) which matrix (and inverse) we want to return
	if M_op==0: # average matrix
		M_ret, m_inv_ret = M_avg, M_inv_avg  
	else: # matrix of straight_lines1 or straight_lines2
		M_ret, m_inv_ret = M[M_op-1], M_inv[M_op-1]
	
	return M_ret, m_inv_ret


### calc_curve_offset: calculate the curvature and the offset of the car from the middle of the lanes 
###   Input: 
###		img_shape: the image shape
###		left_fit: the left lane polynomial estimation
###		right_fit: the right lane polynomial estimation
###		M_inv: the inverse of the birdeye transform matrix 
###		plot_en: plot the original image with the polynomial fit, curvature and offset 
###
###   Output: 
###       left_curverad: the curvature estimation for the left lane (in Km)
###       right_curverad: the curvature estimation for the right lane (in Km)
###       offset: offset of the car from the middle of the lanes (in cm)
def calc_curve_offset(img_shape, left_fit, right_fit, M_inv=None, plot_en=False):
	if left_fit is None:  # we dont have a valid polyfit
		return 0.0, 0.0, 0.0
	
	y_eval = img_shape[1]-1                                                         # y value (nearest to the car) we will calculate the curvature and offset 
	left_lane_org_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]      # x value (in pixels) of the left lane at y_eval
	right_lane_org_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]  # x value (in pixels) of the right lane at y_eval
	lane_width_pix = right_lane_org_x-left_lane_org_x                               # finding the lane width in pixels
	
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/img_shape[1]     # meters per pixel in y dimension
	xm_per_pix = 3.7/lane_width_pix  # meters per pixel in x dimension
	
	# find polynomial in the real world space
	scale_vec = np.float32([xm_per_pix/(ym_per_pix**2), xm_per_pix/ym_per_pix, xm_per_pix])
	left_fit_cr = left_fit*scale_vec
	right_fit_cr = right_fit*scale_vec

	# first we find the curvature.
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	left_curverad /= 1000.0 # switching from meters to km
	right_curverad /= 1000.0 # switching from meters to km

	
	# We now find the offset. 
	offset = ((left_lane_org_x+right_lane_org_x)/2.0 - img_shape[0]/2.0) # assume the camera in in the middle of the car
	offset *= (100.0*xm_per_pix) # switching from pixels to cm
	return left_curverad, right_curverad, offset # switching from meters to km for the curve
	
	
### is_good_lanes: fit a polynomial to the binary birdeye image we get
###   Input: 
###		img_shape: image shape
###		left_fit: the left lane polynomial estimation 
###		right_fit: the right lane polynomial estimation
###
###   Output: 
###       detected: If we think this is a good estimation
###       detected_case: what was the reason for not enabling to detected a good line
def is_good_lanes(img_shape, left_fit, right_fit):
	if left_fit is None:
		return False, 1 # We were not able to fit any line (no points were found)
	
	y_eval = img_shape[1]-1                                                         # y value (nearest to the car) we will calculate the curvature and offset 
	left_lane_org_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]      # x value (in pixels) of the left lane at y_eval
	right_lane_org_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]  # x value (in pixels) of the right lane at y_eval
	left_curverad, right_curverad, _ = calc_curve_offset(img_shape, left_fit, right_fit)
	
	# Conditions that indicate bad estimation 
	if ((left_lane_org_x > 400) or (left_lane_org_x < 200)):    # wrong position of left line near the car
		return False, 2
	if ((right_lane_org_x > 1200) or (right_lane_org_x < 1000)): # wrong position of right line near the car
		return False, 3
	if ((left_curverad > 15.0) or (left_curverad < 0.1)):        # wrong left curve
		return False, 4	
	if ((right_curverad > 15.0) or (right_curverad < 0.1)):      # wrong right curve
		return False, 5		
	
	return True, 0
	
	
### fit_lane_line: fit a polynomial to the binary birdeye image we get
###   Input: 
###		bird_b_undist_img: the input image
###		startover: indicate if we will use the left_fit_prev, right_fit_prev for our search zone
###		left_fit_prev: the left lane polynomial estimation (usually from previous frames) we will use to look for this frame polynomial. if None we will start from scratch 
###		right_fit_prev: the right lane polynomial estimation (usually from previous frames) we will use to look for this frame polynomial. 
###		plot_en: plot the original image and the fit
###
###   Output: 
###       left_fit: the coefficients for the 2ed order polynomial of the left lane
###       right_fit: the coefficients for the 2ed order polynomial of the right lane
###       detected: we were able to detected good lanes on this frame
###       detected_case: what was the reason for not enabling to detected a good line
def fit_lane_line(bird_b_undist_img, startover=True, left_fit_prev=None, right_fit_prev=None, plot_en=False):
	#print('---> Start Fit Polynomial')
	
	nonzero = bird_b_undist_img.nonzero() # Identify the x and y positions of all nonzero pixels in the image
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	out_img = np.dstack((bird_b_undist_img, bird_b_undist_img, bird_b_undist_img))*255 # Create an output image to draw on and visualize the result
	margin = 150                          # Set the width of the windows/poly +/- margin

	if startover:
		#print('\t--> Start over')
		# 1) we find the starting point of the lanes. We also reate an output image to draw on and  visualize the result
		histogram = np.sum(bird_b_undist_img[bird_b_undist_img.shape[0]>>1:,:], axis=0)    # Take a histogram of the bottom half of the image
		midpoint = np.int(histogram.shape[0]>>1)
		leftx_base = np.argmax(histogram[:midpoint])              # Find the peak of the left half of the histogram
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint  # Find the peak of the left half of the histogram

		# 2) define the sliding window and initialize two vectors (x,y) with all the nonzeros values
		nwindows = 9                                                # number of sliding windows
		window_height = np.int(bird_b_undist_img.shape[0]/nwindows) # Set height of windows

		
		# 3) We initialze the strating positions and global values for our scanning 
		minpix_per = 0.3               # Set the percentage of minimum number of pixels found to recenter window
		minpix = int((minpix_per*window_height*margin*2)/100)    # Set the minimum number of pixels found to recenter window
		left_lane_inds = []            # Create empty lists to receive left lane pixel indices
		right_lane_inds = []           # Create empty lists to receive right lane pixel indices
		leftx_current = leftx_base     # Current positions to be updated for each window. left lane
		rightx_current = rightx_base   # Current positions to be updated for each window. right lane
		
		
		# 4) we start step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = bird_b_undist_img.shape[0] - (window+1)*window_height
			win_y_high = bird_b_undist_img.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			
			# Draw the windows on the visualization image
			cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
			cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
			
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
			
			# If you found > minpix pixels, recenter next window on their mean position and add this indications to the lane (this is not just noise)
			if len(good_left_inds) > minpix:   # TODO
				left_lane_inds.append(good_left_inds)    # Append these indices to the lists
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix: 
				right_lane_inds.append(good_right_inds)	 # Append these indices to the lists	
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
		if (len(left_lane_inds) and len(right_lane_inds)):
			# Concatenate the arrays of indices
			left_lane_inds = np.concatenate(left_lane_inds) 
			right_lane_inds = np.concatenate(right_lane_inds)
	else: # use input polynomial
		#print('\t--> Use Previuos Polynomial')
		left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) & (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin))) 
		right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) & (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))  


	left_fit = None
	right_fit = None
	if (len(left_lane_inds) and len(right_lane_inds)):
		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 

		if (len(leftx) and len(rightx)):
			# Fit a second order polynomial to each
			left_fit = np.polyfit(lefty, leftx, 2)
			right_fit = np.polyfit(righty, rightx, 2)


	if (plot_en and left_fit is not None):
		plt.figure(figsize=(16,8))
		# Generate x and y values for plotting
		ploty = np.linspace(0, bird_b_undist_img.shape[0]-1, bird_b_undist_img.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
		result = out_img
		
		if startover==False:
			# Generate a polygon to illustrate the search window area
			# And recast the x and y points into usable format for cv2.fillPoly()
			window_img = np.zeros_like(out_img)
			left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
			left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
			left_line_pts = np.hstack((left_line_window1, left_line_window2))
			right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
			right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
			right_line_pts = np.hstack((right_line_window1, right_line_window2))
			
			cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
			cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
			result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
		
		plt.imshow(result)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.title('Polynomial Fit')
		plt.xlim(0, bird_b_undist_img.shape[1])
		plt.ylim(bird_b_undist_img.shape[0], 0)
		
		if startover:
			plt.savefig('output_images/fit_polynomial_startover.png')
		else:
			plt.savefig('output_images/fit_polynomial_use_prev.png')
	
	detected, detected_case = is_good_lanes((bird_b_undist_img.shape[1],bird_b_undist_img.shape[0]), left_fit, right_fit)
	return left_fit, right_fit, detected, detected_case
	
	
### draw_lane_lines: calculate the curvature and the offset of the car from the middle of the lanes 
###   Input: 
###		rgb_undist_img: the undistorted image
###		left_fit: the left lane polynomial estimation
###		right_fit: the right lane polynomial estimation
###		M_inv: the inverse of the birdeye transform matrix 
###		plot_en: plot the original image with the polynomial fit, curvature and offset 
###
###   Output: 
###       result: the image with the lane lines estimation
def draw_lane_lines(rgb_undist_img, left_fit, right_fit, M_inv=None, plot_en=False):
	#print('---> Start Curvature and Offset Calculation')
	
	img_shape = (rgb_undist_img.shape[1], rgb_undist_img.shape[0])                  # image shape (Xmax, Ymax)
	curve_left, curve_right, offset = calc_curve_offset(img_shape, left_fit, right_fit, M_inv=None, plot_en=False)
	ploty = np.linspace(0, img_shape[1]-1, img_shape[1])                            # Generate y values for calculation

	# Create an image to draw the lines on
	color_warp = np.zeros_like(rgb_undist_img).astype(np.uint8)
	
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, M_inv, img_shape) 
	# Combine the result with the original image
	result = cv2.addWeighted(rgb_undist_img, 1, newwarp, 0.3, 0)
	
	curve = (curve_left+curve_right)/2.0 # average the curve
	cv2.putText(result,'curve = {:.2f} Km'.format(curve),(500,img_shape[1]-80), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
	cv2.putText(result,'offset = {:.1f} cm'.format(offset),(500,img_shape[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
		
	if plot_en:
		plt.figure(figsize=(16,8))
		plt.title('Founded Lane')
		plt.imshow(result)	
		plt.savefig('output_images/final_est_lane.png')
	
	return result
