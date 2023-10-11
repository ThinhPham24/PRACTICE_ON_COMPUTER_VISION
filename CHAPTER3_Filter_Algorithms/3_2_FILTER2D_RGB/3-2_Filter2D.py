# OpenCV provides mainly four types of blurring techniques. Image Bluring (Image Smoothing)
# 1. Average Filtering   --> cv2.boxFilter()
# 2. Gaussian Blurring   --> cv2.GaussianBlur()
# 3. Median Blurring     --> cv2.medianBlur() 
# 4. Bilateral Filtering --> cv2.bilateralFilter() 
# filter2D() uses conv2 the 2 dimensional convolution function to implement filtering operation.
# It is used for blurring, sharpening, embossing, edge detection, and so on. 
# cv2.filter2D(image, -1, kernel) apply the convolution kernel using filter2D function.

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

	img_path = "C:\\Users\\ptthi\\OneDrive\\Desktop\\PRACTICE_ON_COMPUTER_VISION\\CHAPTER3_Filter_Algorithms\\3_2_FILTER2D_RGB\\turtle_noise.png"
	img = cv2.imread(img_path, 1) # 1 provides taking img as rgb
	# image_resize = cv2.resize(img, (640,480), cv2.INTER_AREA)
	# img = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	filter1 = np.array(([0, 1, 0], [1, -5, 1], [0, 1, 0]), np.float32) #linear filter
	# output = cv2.filter2D(img, -1, filter1) 
	# # median_filter = cv2.medianBlur(image_resize,3)
	# merge_image = np.concatenate((image_resize,output), axis=1)
	# cv2.imshow("image", merge_image)
	# cv2.waitKey(0)
	filter2 = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]), np.float32)/9 # 3-by-3 neighborhood avarage filter

	filter4 = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), np.float32) # 3-by-3 neighborhood sharpening filter

	gauss33 = np.array(([1, 2, 1], [2, 4, 2], [1, 2, 1]), np.float32)/16  # 3-by-3 neighborhood gaussian filter

	gauss55 = np.array(([1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]), np.float32)/273 # 5-by-5 neighborhood gaussian filter

	filter6 = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), np.float32) # # 3-by-3 neighborhood edge detection filter

	method = [filter1, filter2, filter4, gauss33, gauss55, filter6]
	out_img = []
	out_img.append(img)
	for i, technique in enumerate(method):
		print("Method:", technique)
		output = cv2.filter2D(img, -1, technique) 
		out_img.append(output)
	titles = ['Original Image', 'linear filter', '3x3 avarage filter', '3x3 sharpe filter', 'Gaussian Blur','5x5 Gaussian Blur',"egde filter"]	   

	for i in range(7):
		plt.subplot(3, 3, i+1) 
		plt.imshow(out_img[i])
		plt.title(titles[i])
		plt.xticks([])
		plt.yticks([])

	plt.show()

	



if __name__ == '__main__':
	main()

