##Writeup Template



**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_and_noncar_example.png
[image2]: ./output_images/car_and_Hog_0.png
[image3]: ./output_images/car_and_Hog_1.png
[image4]: ./output_images/car_and_Hog_2.png
[image5]: ./output_images/noncar_and_Hog_0.png
[image6]: ./output_images/noncar_and_Hog_1.png
[image7]: ./output_images/noncar_and_Hog_2.png
[image8]: ./output_images/all_windows.png
[image9]: ./output_images/test1_car_est.png
[image10]: ./output_images/test2_car_est.png
[image11]: ./output_images/test3_car_est.png
[image12]: ./output_images/test4_car_est.png
[image13]: ./output_images/test5_car_est.png
[image14]: ./output_images/test6_car_est.png
[image15]: ./output_images/video_frames_941_945.png
[image16]: ./output_images/video_frames_946_950.png
[image17]: ./output_images/video_frames_951_955.png
[image18]: ./output_images/video_snapshot_941_955.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. [Here](https://github.com/shayko18/CarND-Vehicle-Detection) is a link to my project repository.  

You can find there:

- vehiclesDetection.py: this is the main python file for this project
- advancedLaneLines_funcs.py: this is the main python file from the previous project (Advanced-Lane-Lines). We'll use its functions also here
- project_video_with_cars.mp4: the video output of this project
- writeup_report.md: the writeup of this project
- output_images folder: images that we will use in this writeup

###Overview of the main python file
We divide the file to five parts:

- Part A (Lines 22-520): This part hold all the help functions we will use. Most of them are the function from class and there are some new ones that are mainly used to plot the images we will use in this writeup.
- Part B (Lines 522-615):This part is used to set the parameters we will use to extract the features and to train and test the SVC model. After we find the SVC model we save the relevant parameters to "svc_data.p". There is the option to skip training the model each time but to load the saved data (using: load_svc=True)
- Part C (Lines 617-650): We set parameters that we will use in the main pipeline:
	- The windows that we will use in the sliding window (we also plot them)
	- The SVC model and the parameters we will use to extract the features 
	- Some data from the previous project (Advanced-Lane-Lines) we will use to also plot the lane lines here
- Part D (Lines 653-776): The main pipeline. Only one function- process_image(). we will use it both for the video and for the single test images.
- Part E (Lines 778-798): Here we decide if the run the pipeline on a single image (and which image) and/or on the video (and which video)

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In lines 541-565 we extracted the features from the given images of the `vehicle` and `non-vehicle`. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
 
![alt text][image1]

Next for each image we extracted the features we want. I used the spatial features, the histogram features and the most important is the HOG features. The function that was used for it was extract_features(...) (Line 95-166). This function was talked about in class. To get the HOG features it calls get_hog_features(...) (Line 39-64) which will simply call the hog function.


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. I also tested some configurations with the SVC to see what are the result I get. When I added also the spatial features and the histogram features the SVC estimator improved from about 95% to 98%. Of course it will require more computation but I decided to stay with it for now.

Here is an example using the `YCrCb` color space and HOG parameters fron the Y-channel of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. The first three examples are for car images and the last three are for non car images:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I checked the performance I get from the SVC estimator. Of course, there is some tradeoff with the number of features, but the values we set seems reasonable and gave good performance: error rate of about 2%
It also seemed a good idea to use the Y channel (gray scale) to get the HOG features, and that is what I did.
Those are the final parameters I used (including the other features) (Lines 528-539):

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

Total number of features:

	Feature vector length: 2580


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

We read all the images that are given to us of cars and non-cars. We can see that they are balanced more or less:

	Vehicles images length = 8792
	Non-Vehicles images length = 8968

After extracting and scaling all the features we have we took few steps to train the SVC model:

- shuffle all the data we have (Line 570-571)
- split the data to training set (80%) and test set (20%) (Lines 571)
	- Train data length = 14208
	- Test data length = 3552

- fit a linear SVC model (Lines 581-585)
- test the SVC model (Line 586)
	7.63 Seconds to train SVC...
	- Test Accuracy of SVC =  0.9809


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The windows are set in the function set_search_windows(...) (Lines 356-393)
- Window sizes: As was talked about in class we wanted different window sizes because the different size of the car (in pixels) as it is near to us or far away from us - I took 4 sizes.
- Global search area: The search area should be only "on the road level" (not in the sky) and this is why all windows search area in limited to y_value_start of 400 
- Search area per window: Again, as was talked about in class there is no reason to look for "small cars" near us, this is why each window stops searching at the different y_value_stop.
- For the x-axis we search in all the pixels (we don't want to assume the lane we are driving on)
- overlapping: There is a big tradeoff between the performance and the computation power. With big overlapping we will get big number of windows, the performance (especially when we will use some simple way to remove false detections) should be better but the computation power will be also big. I set xy_overlap to 0.75 for all the windows as I saw it gave me good performance.

Here are all the windows we will use:

- Blue: y_start_stop=[400, 656], xy_window=(128, 128), xy_overlap=(0.75, 0.75)
- Green: y_start_stop=[400, 592], xy_window=(96, 96), xy_overlap=(0.75, 0.75)
- Red: y_start_stop=[400, 528], xy_window=(64, 64), xy_overlap=(0.75, 0.75)
- Pink: y_start_stop=[400, 496], xy_window=(48, 48), xy_overlap=(0.75, 0.75)

![alt text][image8]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As I talked before I tried different features configuration, different color maps, different HOG parameters and different windows. In the following figures we can see for a few test images:

-	how the SVC model did: You can see here some false detections. In order to remove those detections we used the heatmap algorithm that was talked about in class. The heatmap threshold I used was 5 (heat_map_th, Line 668)
-	Heatmap per image after thresholding that help us remove false detections.
-	Final estimation per image. We can see better performance than what only the SVC has done 
 

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/shayko18/CarND-Vehicle-Detection/blob/master/project_video_with_cars.mp4)

I aslo added the lane lines from the last project.

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The pipeline for each frame is in the function process_image(...) (Lines 653-776). The stages in the pipeline:

- Each frame is passed in the SVC using the function search_windows(...) (Lines 735-740). and we get all the detected windows on this frame.
- We try to filter out false detection on this frame using the heatmap on it, using a configurable threshold (I set it to 5) (Lines 743-747)
- We want also to filter the results over time (frames), so we save the last `n_frames_history` frames heatmap (set it to 15) (Lines 749-751)
- To get the estimation on the current frame we create a heatmap_over_time using all the last heatmaps we saved. (Lines 752)
- I than threshold this new map using a new threshold which is the single frame threshold times the number of history heatmaps (Lines 753-754)- 
- I then used `scipy.ndimage.measurements.label()` to identify individual blobs in this heatmap_over_time and we assume each blob corresponded to a vehicle. (Lines 756)
- I constructed bounding boxes to cover the area of each blob detected (Lines 757)
  

Here's an example result showing the heatmap from a series of frames of video (frames 941-955), the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 15 frames and their corresponding heatmaps (**just using this frame - no history**):

![alt text][image15]
![alt text][image16]
![alt text][image17]

### Here is the output of `label()` on the integrated heatmap from all 15 frames and the bounding boxes are drawn onto the 955th frame:
![alt text][image18]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


- One issue is that are a lot of parameters to configurated. I think that in a different video the configuration I selected (which can take some time to optimize) would not work as good as in this video. 
-	There is a question of performance vs computation power. We didn’t payed much emphasis on the computation power in this project. There are a few ways we could improve this – for example not to search every frame entire area but to look only on the hot areas we have so far and only once every few frame to search the entire area.
-	There is also a trade of on the filter BW we used on the frames. long history we remove false detections but could be less accurate in the car position and it will take it more time to mark a new car
-	I added also the lane line estimation from the previous project.
 

