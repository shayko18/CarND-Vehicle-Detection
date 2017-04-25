import cv2
import glob
import time
import math
import pickle
import numpy as np
import random
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from PIL import Image
from moviepy.editor import VideoFileClip
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from advancedLaneLines_funcs import *


####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part A: Help Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
### draw_boxes: plots on an image, a list of bounding boxes
###   Input: 
###       img: input image
###       bboxes: bounding boxes looks like this = [((,), (,)), .... ((,), (,)), ... ]
###       color: line color in RGB format
###       thick: line thickness 
###
###   Output: 
###      draw_img: output image with the boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	draw_img = np.copy(img) # Make a copy of the image
	for bbox in bboxes:     # Iterate through the bounding boxes
		cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick) # Draw a rectangle given bbox coordinates
	return draw_img         # Return the image copy with boxes drawn


### get_hog_features: Define a function to return HOG features and visualization
###   Input: 
###       img: input image
###       orient: number of orientations
###       pix_per_cell: pixels per cell
###       cell_per_block: cell per block  
###       vis: visitation enabling   
###       feature_vec: return feature vector   
###
###   Output: 
###      features: return feature
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):                
	if vis == True: # Call with two outputs if vis==True
		features, hog_image = hog(img, orientations=orient, 
								  pixels_per_cell=(pix_per_cell, pix_per_cell),
								  cells_per_block=(cell_per_block, cell_per_block), 
								  transform_sqrt=True, 
								  visualise=vis, feature_vector=feature_vec)
		return features, hog_image
	else:           # Otherwise call with one output
		features = hog(img, orientations=orient, 
					   pixels_per_cell=(pix_per_cell, pix_per_cell),
					   cells_per_block=(cell_per_block, cell_per_block), 
					   transform_sqrt=True, 
					   visualise=vis, feature_vector=feature_vec)
		return features


### bin_spatial: Define a function to compute binned color features
###   Input: 
###       img: input image
###       size: size of wanted image to calculate the features  
###
###   Output: 
###      features: return feature 
def bin_spatial(img, size=(32, 32)):
	features = cv2.resize(img, size).ravel() # Use cv2.resize().ravel() to create the feature vector
	return features                          # Return the feature vector


### color_hist: Define a function to compute binned color features
###   Input: 
###       img: input image
###       nbins: number of bins 
###       bins_range: bins range  
###
###   Output: 
###      hist_features: return feature 
def color_hist(img, nbins=32, bins_range=(0, 1)):
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)                   # Compute the histogram of the color channels separately
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))   # Concatenate the histograms into a single feature vector
	return hist_features  # Return the individual histograms, bin_centers and feature vector


### extract_features: Define a function to extract features from a list of images
###   Input: 
###       imgs: input images
###       color_space: color space to work on
###       spatial_size: spatial size  
###       hist_bins: number of bins  
###       orient: number of orientations (for hog)  
###       pix_per_cell: pix per cell (for hog)  
###       cell_per_block: cell per block (for hog)  
###       hog_channel: hog channel to use (for hog)  
###       spatial_feat: enable spatial features  
###       spatial_feat: enable spatial features  
###       hist_feat: enable color hist features  
###       hog_feat: enable hog features  
###
###   Output: 
###      features: return all features
###      channel_image: hog image example list
###      hog_image: hog image example list
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, plot_example=False):
	channel_image = [] # set to empty array if we will not use it
	hog_image = []	   # set to empty array if we will not use it
	features = []      # Create a list to append feature vectors to
	for file in imgs:  # Iterate through the list of images
		file_features = []
		image = mpimg.imread(file) # Read in each one by one
		# apply color conversion if other than 'RGB'
		if color_space != 'RGB':
			if color_space == 'HSV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif color_space == 'LUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif color_space == 'HLS':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif color_space == 'YUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif color_space == 'YCrCb':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
		else: feature_image = np.copy(image)      

		if spatial_feat == True:
			spatial_features = bin_spatial(feature_image, size=spatial_size)
			file_features.append(spatial_features)
		if hist_feat == True:
			# Apply color_hist()
			hist_features = color_hist(feature_image, nbins=hist_bins)
			file_features.append(hist_features)
		if hog_feat == True:
		# Call get_hog_features() with vis=False, feature_vec=True
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,channel], 
										orient, pix_per_cell, cell_per_block, 
										vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)        
			else:	
				if (plot_example):
					hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
					channel_image = feature_image[:,:,hog_channel]
				else:
					hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
				
			# Append the new feature vector to the features list
			file_features.append(hog_features)
		features.append(np.concatenate(file_features))
		if (plot_example):
			break
	return features, channel_image, hog_image # Return list of feature vectors
    

### slide_window: Define the sliding window function
###   Input: 
###       shape: input image shape
###       x_start_stop: start and stop positions on x axis
###       y_start_stop: start and stop positions on y axis
###       xy_window: window size (x and y dimensions)  
###       xy_overlap: overlap fraction (for both x and y) 
###
###   Output: 
###      window_list: list of all the windows
def slide_window(shape, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = shape[0]
	# Compute the span of the region to be searched    
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	# Compute the number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
	# Compute the number of windows in x/y
	nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
	ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
	nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
	ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
	# Initialize a list to append window positions to
	window_list = []
	# Loop through finding x and y window positions
	# Note: you could vectorize this step, but in practice
	# you'll be considering windows one by one with your
	# classifier, so looping makes sense
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			
			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
	# Return the list of windows
	return window_list


### single_img_features: Define a function to extract features from a single image window
###   Input: 
###       img: input image
###       the rest is same as in extract_features(...)
###
###   Output: 
###      image features
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
	#1) Define an empty list to receive features
	img_features = []
	#2) Apply color conversion if other than 'RGB'
	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else: feature_image = np.copy(img)      
	#3) Compute spatial features if flag is set
	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		#4) Append features to list
		img_features.append(spatial_features)
	#5) Compute histogram features if flag is set
	if hist_feat == True:
		hist_features = color_hist(feature_image, nbins=hist_bins)
		#6) Append features to list
		img_features.append(hist_features)
	#7) Compute HOG features if flag is set
	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend(get_hog_features(feature_image[:,:,channel], 
									orient, pix_per_cell, cell_per_block, 
									vis=False, feature_vec=True))      
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		#8) Append features to list
		img_features.append(hog_features)

	#9) Return concatenated array of features
	return np.concatenate(img_features)


### search_windows: Define a function you will pass an image and the list of windows to be searched (output of slide_windows())
###   Input: 
###       img: input image
###       the rest are similar to single_img_features(...)
###
###   Output: 
###      on_windows: windows with a hit (detect a car)
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 1), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
	#1) Create an empty list to receive positive detection windows
	on_windows = []
	#2) Iterate over all windows in the list
	for window in windows:
		#3) Extract the test window from original image
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
		#4) Extract features for that window using single_img_features()
		features = single_img_features(test_img, color_space=color_space, 
							spatial_size=spatial_size, hist_bins=hist_bins, 
							orient=orient, pix_per_cell=pix_per_cell, 
							cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_feat=spatial_feat, 
							hist_feat=hist_feat, hog_feat=hog_feat)
		#5) Scale extracted features to be fed to classifier
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		#6) Predict using your classifier
		prediction = clf.predict(test_features)
		#7) If positive (prediction == 1) then save the window
		if prediction == 1:
			on_windows.append(window)
	#8) Return windows for positive detections
	return on_windows


### add_heat: Define a function that create a hit map
###   Input: 
###       heatmap: blank (all zeros) input 
###       bbox_list: windows that got a hit
###
###   Output: 
###      heatmap: the final heat map 
def add_heat(heatmap, bbox_list):
	for box in bbox_list: # Iterate through list of bboxes
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	return heatmap # Return updated heatmap


### apply_threshold: Define a function that applys threshold on the heatmap
###   Input: 
###       heatmap: heatmap input 
###       threshold: threshold for the heatmap
###
###   Output: 
###      heatmap: the final heat map after thresholding it
def apply_threshold(heatmap, threshold):
	heatmap[heatmap <= threshold] = 0 # Zero out pixels below the threshold
	return heatmap # Return thresholded map


### draw_labeled_bboxes: Define a function that draws the estimated cars position
###   Input: 
###       img: input image 
###       labels: labels for the cars
###
###   Output: 
###      img: the final image with the estimated cars positions
def draw_labeled_bboxes(img, labels):
	for car_number in range(1, labels[1]+1):               # Iterate through all detected cars
		nonzero = (labels[0] == car_number).nonzero()      # Find pixels with each car_number label value
		nonzeroy = np.array(nonzero[0])                    # Identify x and y values of those pixels
		nonzerox = np.array(nonzero[1])
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))) # Define a bounding box based on min/max x and y
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6) # Draw the box on the image
	return img # Return the image


### set_search_windows: Define a function that draws the estimated cars position
###   Input: 
###       is_draw: If we want to draw all the windows 
###
###   Output: 
###      windows: all the windows
def set_search_windows(is_draw=False):
	img_fname_tmp = 'test_images/test1.jpg'   # image name
	image_tmp = cv2.imread(img_fname_tmp)          # read the image
	image_tmp = cv2.cvtColor(image_tmp,cv2.COLOR_BGR2RGB) # convert to rgb
	shape = image_tmp.shape
	windows = slide_window(shape, x_start_stop=[None, None], y_start_stop=[400, 656], xy_window=(128, 128), xy_overlap=(0.75, 0.75))
	if is_draw:
		window_img = draw_boxes(image_tmp, windows, color=(0, 0, 255), thick=5) # draw all the boxes we found
	
	windows_tmp = slide_window(shape, x_start_stop=[None, None], y_start_stop=[400, 592], xy_window=(96, 96), xy_overlap=(0.75, 0.75))
	windows += windows_tmp
	if is_draw:
		window_img = draw_boxes(window_img, windows_tmp, color=(0, 255, 0), thick=4) # draw all the boxes we found
	
	windows_tmp = slide_window(shape, x_start_stop=[None, None], y_start_stop=[400, 528], xy_window=(64, 64), xy_overlap=(0.75, 0.75))
	windows += windows_tmp
	if is_draw:
		window_img = draw_boxes(window_img, windows_tmp, color=(255, 0, 0), thick=3) # draw all the boxes we found
	
	windows_tmp = slide_window(shape, x_start_stop=[None, None], y_start_stop=[400, 496], xy_window=(48, 48), xy_overlap=(0.75, 0.75))
	windows += windows_tmp
	if is_draw:
		window_img = draw_boxes(window_img, windows_tmp, color=(255, 0, 255), thick=2) # draw all the boxes we found
	
	if is_draw:
		plt.figure(figsize=(16,8))
		plt.imshow(window_img)
		plt.title('All Sliding Windows we use')
		plt.savefig("output_images/all_windows.png")
	
	return windows

	
### plot_car_noncar_example: Define a function that draws an example for a car and noncar images
###   Input: 
###       car_img_name: car/noncar image file name 
###       non_car_img_name: car/noncar image file name 
###
###   Output: 
###   
def plot_car_noncar_example(car_img_name, non_car_img_name):	
	img_car = mpimg.imread(car_img_name)
	img_noncar = mpimg.imread(non_car_img_name)
	plt.figure(figsize=(16,8))
	plt.subplot(1,2,1)
	plt.imshow(img_car)
	plt.title('Car')

	plt.subplot(1,2,2)
	plt.imshow(img_noncar)
	plt.title('Non-Car')
	
	plt.savefig("output_images/car_and_noncar_example.png")

		
### plot_hog_example: Define a function that draws HOG image
###   Input: 
###       is_car: is it a car image 
###       img_name: car/noncar image file name 
###       feature_img: the channel we used to get the HOG image 
###       hog_img: hog image of the car 
###       idx: number to make sure we name the figures differently  
###
###   Output: 
###   
def plot_hog_example(is_car, img_name, channel_img, hog_img, idx):	
	img = mpimg.imread(img_name)
	plt.figure(figsize=(16,8))
	plt.subplot(1,3,1)
	plt.imshow(img)
	if (is_car):
		plt.title('Original Car Image (idx={})'.format(idx))
	else:
		plt.title('Original NonCar Image (idx={})'.format(idx))

	plt.subplot(1,3,2)
	plt.imshow(channel_img, cmap='gray')
	plt.title('Y channel (idx={})'.format(idx))
	
	plt.subplot(1,3,3)
	plt.imshow(hog_img, cmap='gray')
	plt.title('Hog Image (idx={})'.format(idx))
	
	if (is_car):
		plt.savefig("output_images/car_and_Hog_"+ str(idx) +".png")
	else:
		plt.savefig("output_images/noncar_and_Hog_"+ str(idx) +".png")

		
### plot_single_frame: Define a function that draws the estimated cars position for a single frame 
###   Input: 
###       fname: image name
###       img: original image
###       hot_windows: raw SVC estimation 
###       heatmap: after we use the threshold to remove false alarms 
###       draw_img: final estimation of the cars 
###
###   Output: 
###   
def plot_single_frame(fname, img, hot_windows, heatmap, draw_img):	
	plt.figure(figsize=(16,8))
	plt.subplot(2,2,1)
	plt.imshow(img)
	plt.title('Original Image - {}'.format(fname))

	plt.subplot(2,2,2)
	draw_image = np.copy(img)
	window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
	plt.imshow(window_img)
	plt.title('raw SVC estimation')
	
	plt.subplot(2,2,3)
	plt.imshow(heatmap, cmap='hot')
	plt.title('Heat map')
	
	plt.subplot(2,2,4)
	plt.imshow(draw_img)
	plt.title('Final Estimation')
	
	plt.savefig("output_images/"+ fname +"_car_est.png")	

	
### plot_frames_over_time: Define a function that draws the estimated cars position for a single frame 
###   Input: 
###       imgs: images on consecutive frames 
###       heat_over_time: heat map over time - each frame on its on
###       final_label: final label estimation 
###       final_draw: final car position estimation 
###
###   Output: 
###   
def plot_frames_over_time(imgs, heat_over_time, final_label, final_draw, frame_trig):	
	n_frames=len(imgs)
	frames_per_fig = 5
	
	for j in range(np.int(math.ceil(n_frames/frames_per_fig))):
		plt.figure(figsize=(16,8))
		for i in range(frames_per_fig):
			k = j*frames_per_fig+i
			plt.subplot(2,frames_per_fig,1+i)
			plt.imshow(imgs[k])
			plt.title('Frame, k={}'.format(frame_trig+k))
			
			plt.subplot(2,frames_per_fig,1+i+frames_per_fig)
			heatmap = np.clip(heat_over_time[:,:,-1-k], 0, 255)
			plt.imshow(heatmap, cmap='hot')
			plt.title('Heat map per frame, k={}'.format(frame_trig+k))
		plt.savefig("output_images/video_frames_"+str(frame_trig+j*frames_per_fig)+"_"+str(frame_trig+(j+1)*frames_per_fig-1)+".png")
	
	plt.figure(figsize=(16,8))
	plt.subplot(1,2,1)
	plt.imshow(final_label[0], cmap='gray')
	plt.title('Final Label est, frames {} to {}'.format(frame_trig,frame_trig+n_frames-1))
	plt.subplot(1,2,2)
	plt.imshow(final_draw)
	plt.title('Final est, frames {} to {}'.format(frame_trig,frame_trig+n_frames-1))
	
	plt.savefig("output_images/video_snapshot_"+str(frame_trig)+"_"+str(frame_trig+n_frames-1)+".png")
	

####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part B: Train the SVC model and Set the feature parametres ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
load_svc = False
pickle_file_svc = 'svc_data.p'  
pickle_file_preprocess = 'preprocess_data.p'
if (load_svc==False):
	debug_len = -1 
		
	color_space = 'YCrCb'   # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9              # HOG orientations
	pix_per_cell = 8        # HOG pixels per cell
	cell_per_block = 2      # HOG cells per block
	hog_channel = 0         # Can be 0, 1, 2, or "ALL"
	spatial_size = (16, 16) # Spatial binning dimensions
	hist_bins = 16          # Number of histogram bins
	spatial_feat = True     # Spatial features on or off
	hist_feat = True        # Histogram features on or off
	hog_feat = True         # HOG features on or off


	imgs_cars = glob.glob('vehicles/*/*.png')
	if debug_len>0:
		imgs_cars = imgs_cars[0:debug_len]	
	
	car_features, _, _ = extract_features(imgs_cars, color_space=color_space, 
							spatial_size=spatial_size, hist_bins=hist_bins, 
							orient=orient, pix_per_cell=pix_per_cell, 
							cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_feat=spatial_feat, 
							hist_feat=hist_feat, hog_feat=hog_feat)
							
	imgs_non_cars = glob.glob('non-vehicles/*/*.png')
	if debug_len>0:
		imgs_non_cars = imgs_non_cars[0:debug_len]
	notcar_features, _, _ = extract_features(imgs_non_cars, color_space=color_space, 
							spatial_size=spatial_size, hist_bins=hist_bins, 
							orient=orient, pix_per_cell=pix_per_cell, 
							cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_feat=spatial_feat, 
							hist_feat=hist_feat, hog_feat=hog_feat)

		
	X = np.vstack((car_features, notcar_features)).astype(np.float64)                        	
	X_scaler = StandardScaler().fit(X) # Fit a per-column scaler	
	scaled_X = X_scaler.transform(X) # Apply the scaler to X

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
	
	print('Vehicles images length = {}'.format(len(imgs_cars)))
	print('Non-Vehicles images length = {}'.format(len(imgs_non_cars)))
	print('Train data length = {}'.format(len(X_train)))
	print('Test data length = {}'.format(len(X_test)))

	print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')	
	print('Feature vector length:', len(X_train[0]))
	 
	svc = LinearSVC() # Use a linear SVC
	t=time.time()     # Check the training time for the SVC
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4)) # Check the score of the SVC
	# Save the data for easy access
	with open(pickle_file_svc, 'wb') as pfile:
		pickle.dump({'color_space': color_space, 'orient': orient, 'pix_per_cell': pix_per_cell, 'cell_per_block': cell_per_block, 'hog_channel': hog_channel,
					 'spatial_size': spatial_size, 'hist_bins': hist_bins, 'spatial_feat': spatial_feat, 'hist_feat': hist_feat, 'hog_feat': hog_feat, 'X_scaler': X_scaler, 'svc': svc}, pfile)

	####
	### Plot some examples. Car and non-car and HOG with cars and noncars
	idxc = np.random.randint(0, (len(imgs_cars)-1))
	idxn = np.random.randint(0, (len(imgs_non_cars)-1))
	plot_car_noncar_example(imgs_cars[idxc], imgs_non_cars[idxn])
	for j in range(3):	
		idxc = np.random.randint(0, (len(imgs_cars)-2))
		features_ex, img_ex, hog_img_ex = extract_features(imgs_cars[idxc:idxc+1], color_space=color_space, 
							spatial_size=spatial_size, hist_bins=hist_bins, 
							orient=orient, pix_per_cell=pix_per_cell, 
							cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_feat=spatial_feat, 
							hist_feat=hist_feat, hog_feat=hog_feat, plot_example=True)
		plot_hog_example(True, imgs_cars[idxc], img_ex, hog_img_ex, j)
		
		idxn = np.random.randint(0, (len(imgs_non_cars)-2))
		features_ex, img_ex, hog_img_ex = extract_features(imgs_non_cars[idxn:idxn+1], color_space=color_space, 
							spatial_size=spatial_size, hist_bins=hist_bins, 
							orient=orient, pix_per_cell=pix_per_cell, 
							cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_feat=spatial_feat, 
							hist_feat=hist_feat, hog_feat=hog_feat, plot_example=True)
		plot_hog_example(False, imgs_non_cars[idxn], img_ex, hog_img_ex, j)


####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part C: Parameters and SVC for the Pipeline ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
###     1) Set the windows we will search on
###     2) Load the data we used also in the SVC training
###     3) Load the data we used for the preprocessing image

### 1) Get windows: We used a temp image just to get the image shape
windows = set_search_windows(is_draw=True)

### 2) Load the data we used also in the SVC training
with open(pickle_file_svc, mode='rb') as pfile:
	pfile_data = pickle.load(pfile)

color_space = pfile_data['color_space']
orient = pfile_data['orient']
pix_per_cell = pfile_data['pix_per_cell']
cell_per_block = pfile_data['cell_per_block']
hog_channel = pfile_data['hog_channel']
spatial_size = pfile_data['spatial_size']
hist_bins = pfile_data['hist_bins']
spatial_feat = pfile_data['spatial_feat']
hist_feat = pfile_data['hist_feat']
hog_feat = pfile_data['hog_feat']
X_scaler = pfile_data['X_scaler']
svc = pfile_data['svc']

### 3) Load the data we used for the preprocessing image
with open(pickle_file_preprocess, mode='rb') as pfile:
	pfile_data = pickle.load(pfile)

mtx = pfile_data['mtx']
dist = pfile_data['dist']
M = pfile_data['M']
M_inv = pfile_data['M_inv']



####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part D: Main Pipeline ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
### process_image: This is the main pipeline. we take the input rgb image, estimate the lane lines and draw them on the image 
###   Input: 
###		img: the input RGB image 
###		single_img: single image or video mode
###		fname: relevant for single mode - just to print the image we worked on 
###
###   Output: 
###       result: the image with the lane lines estimation
def process_image(img, single_img=False, fname=''):
	also_lanes = True      # also plot the lane lines
	
	##
	## define function vars - both for the vehicles and lane line detection
	n_frames_history = 15  # history length
	heat_map_th = 5        # heat map threshold for a single frame
	snapshot_trig = 940    # snapshot, for plotting in the writeup 940
	
	a_good = 0.75 # the IIR filter coeff. The weight of the current estimation when detected=True
	a_bad = 0.25  # the IIR filter coeff. The weight of the current estimation when detected=False
	bad_frames_th = 3 # number of consecutive bad frames before we startover in the polyfit estimation
	
	##
	## define static vars - both for the vehicles and lane line detection
	if not hasattr(process_image, "first_frame"):
		process_image.first_frame = True  # it doesn't exist yet, so initialize it
	if not hasattr(process_image, "cnt"):
		process_image.cnt = 0  # it doesn't exist yet, so initialize it
	if not hasattr(process_image, "img_frames_snapshot"):
		process_image.img_frames_snapshot = []  # it doesn't exist yet, so initialize it
	if not hasattr(process_image, "heat_over_time"):
		process_image.heat_over_time = np.zeros_like(img[:,:,0]).astype(np.float) # it doesn't exist yet, so initialize it
		process_image.heat_over_time = np.expand_dims(process_image.heat_over_time, axis=2)
		process_image.heat_over_time = np.repeat(process_image.heat_over_time, n_frames_history, axis=2) # it doesn't exist yet, so initialize it
	
	if not hasattr(process_image, "n_bad_frames"):
		process_image.n_bad_frames = 1000  # it doesn't exist yet, so initialize it
	if not hasattr(process_image, "left_fit"):
		process_image.left_fit = None  # it doesn't exist yet, so initialize it
	if not hasattr(process_image, "right_fit"):
		process_image.right_fit = None  # it doesn't exist yet, so initialize it
	
	if (False and process_image.cnt<(snapshot_trig-2)): # for debug
		process_image.cnt+=1
		return img
	
	##
	## Finding lane lines	
	if (also_lanes==True):
		rgb_undist_img = calibrate_road_image(img, mtx, dist, fname=fname, plot_en=single_img)                       # apply the Calibration parameters on one of the test images
		b_undist_img = apply_binary_th(rgb_undist_img, plot_en=single_img)                                           # b_ stands for binary  
		bird_b_undist_img = cv2.warpPerspective(b_undist_img, M, (rgb_undist_img.shape[1], rgb_undist_img.shape[0])) # apply the matrix
		left_fit_curr, right_fit_curr, detected, detected_case = fit_lane_line(bird_b_undist_img, startover=(process_image.n_bad_frames>bad_frames_th), left_fit_prev=process_image.left_fit, right_fit_prev=process_image.right_fit, plot_en=single_img)   # find the lane line fit

		if process_image.first_frame: # first time ever
			process_image.left_fit = left_fit_curr
			process_image.right_fit = right_fit_curr
			process_image.first_frame = single_img
			if detected:
				process_image.n_bad_frames = 1000
		else:
			if detected:     # the current estimation looks good, we weight it with the previous estimation (IIR)
				a = a_good
				process_image.n_bad_frames = 0
			else:
				a = a_bad
				process_image.n_bad_frames += 1
			
			process_image.left_fit = (1.0-a)*process_image.left_fit + a*left_fit_curr
			process_image.right_fit = (1.0-a)*process_image.right_fit + a*right_fit_curr
		
		result = draw_lane_lines(rgb_undist_img, process_image.left_fit, process_image.right_fit, M_inv=M_inv, plot_en=single_img)    # draw the estimated lane lines, the curvature and the offset
		
	else: 
		result = np.copy(img)
	
	##
	## Finding vehicles
	draw_image = np.copy(result)        # the output image initialization
	img = img.astype(np.float32)/255.0  # transform the image to [0,1) float values

	## Search all windows and return the widows that found a car
	hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space, 
							spatial_size=spatial_size, hist_bins=hist_bins, 
							orient=orient, pix_per_cell=pix_per_cell, 
							cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_feat=spatial_feat, 
							hist_feat=hist_feat, hog_feat=hog_feat)


	## Heat map on the current frame
	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	heat = add_heat(heat,hot_windows)         # Add heat to each box in box list
	heat = apply_threshold(heat,heat_map_th)  # Apply threshold to help remove false positives
	heatmap = np.clip(heat, 0, 255)           # Visualize the heatmap when displaying

	## Heat map on the last 'n_frames_history' frames
	process_image.heat_over_time[:,:,1:] = process_image.heat_over_time[:,:,0:-1]
	process_image.heat_over_time[:,:,0] = heat
	heat_lpf = np.sum(process_image.heat_over_time[:,:,:], axis=2)
	heat_lpf = apply_threshold(heat_lpf,heat_map_th*np.clip(process_image.cnt+1, 0, n_frames_history))  # Apply threshold (n_frames_history * single_frame_th) to help remove false positives
	heatmap_lpf = np.clip(heat_lpf, 0, 255)                                                             # Visualize the heatmap when displaying
	
	labels = label(heatmap_lpf)                                     # Find final boxes from heatmap using label function
	draw_img = draw_labeled_bboxes(np.copy(draw_image), labels)	    # Draw the vehicles location

	if ((process_image.cnt+1)>snapshot_trig and (process_image.cnt+1)<=(n_frames_history+snapshot_trig)):
		process_image.img_frames_snapshot.append(draw_image)
		if ((process_image.cnt+1)==(n_frames_history+snapshot_trig)):
			plot_frames_over_time(process_image.img_frames_snapshot, process_image.heat_over_time, labels, draw_img, snapshot_trig+1)
	
	
	if (False and single_img==False): # for debug
		cv2.imwrite("video_images/img" + str(process_image.cnt) + ".jpg", cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR))	
		
	if single_img:
		plot_single_frame(fname[12:17], draw_image, hot_windows, heatmap, draw_img)
	else:
		process_image.cnt+=1
		if ((process_image.cnt%100)==0):
			print('cnt={}'.format(process_image.cnt))
	
	return draw_img
	
	
####  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Part E: Using the Pipeline on a single image or on a video ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ####
single_img_en = False  # enable single image
video_en = True        # enable the video

###
### Single image
if single_img_en:
	img_fname = 'test_images/test1.jpg'   # image name
	img = cv2.imread(img_fname)           # read the image
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	process_image(img, single_img=True, fname=img_fname)

	
###
### Video: Read the video and add the estimated lane lines on it
if video_en:
	video_fname = 'project_video'   # image name  test_video or project_video
	output = video_fname + '_with_cars.mp4'
	clip = VideoFileClip(video_fname+'.mp4')
	out_clip = clip.fl_image(process_image) 
	out_clip.write_videofile(output, audio=False)