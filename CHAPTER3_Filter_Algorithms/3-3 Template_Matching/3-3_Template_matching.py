import cv2
import numpy as np
from matplotlib import pyplot as plt
image_source = cv2.imread('C:\\Users\\ptthi\\OneDrive\\Desktop\\PRACTICE_ON_COMPUTER_VISION\\CHAPTER3_Filter_Algorithms\\3-3 Template_Matching\\flower.png')
image_source_BGR = image_source
image_source = cv2.cvtColor(image_source_BGR, cv2.COLOR_BGR2GRAY)
image_template = cv2.imread('C:\\Users\\ptthi\\OneDrive\\Desktop\\PRACTICE_ON_COMPUTER_VISION\\CHAPTER3_Filter_Algorithms\\3-3 Template_Matching\\flower_template.png',0)
w, h = image_template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
if __name__ == "__main__":
    
	'''
	#------
	# WAY1: Manual implementation 
	#--------
	S_cols = image_source.shape[1]
	S_rows = image_source.shape[0]
	T_cols = image_template.shape[1]
	T_rows = image_template.shape[0]
	SSD = 1
	positionx = 0
	positiony = 0
	new_feature = np.zeros((int((S_rows-T_rows)/1)+1,int((S_cols-T_cols)/1)+1)) 
	for h in range(int((S_rows-T_rows)+1)):
		for w in range(int((S_cols-T_cols)+1)):
			A = image_source[h:h + (T_rows), w:w + (T_cols)]
			B = image_template
			# Calculate cross - corrlelation
			cross = ((image_source[h:h + (T_rows), w:w + (T_cols)])*image_template)
			E = np.sum(cross)
			# Normalize I(x+x',y+y') and I(x',y')
			C = np.linalg.norm(A,ord='fro')
			D = np.linalg.norm(B,ord='fro')
			# Normalized Cross - Correlation
			Ncc = E/(C*D)
			# Comparation
			if Ncc < SSD:
				positionx = w 
				positiony = h 
				SSD = Ncc
			# Draw rectangle
			img_S = image_source.astype(np.uint8)
			cv2.rectangle(img_S, (int(w), int(h)), (int((w + T_cols)), int((h + T_rows))), (255, 0, 0), 1)
			# cv2.rectangle(img_S, (int(w+STRIDE-1), int(h+STRIDE-1)), (int((w+STRIDE-1 + KERNAL_SIZE)), int((h+STRIDE-1 + KERNAL_SIZE))), (255, 0, 0), 1)
			cv2.namedWindow('Conv_process', cv2.WINDOW_NORMAL)
			cv2.resizeWindow("Conv_process", 300, 300)
			cv2.imshow('Conv_process',img_S)
			cv2.waitKey(1)

	(x1,y1) = (positionx, positiony)
	(x2,y2) = (positionx + T_cols, positiony + T_rows)
	cv2.rectangle(image_source_BGR, (int(x1), int(y1)), (x2, y2), (255, 255, 0), 2)
	cv2.namedWindow('Conv_result', cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Conv_result", 300, 300)
	cv2.imshow('Conv_result',image_source_BGR)
	cv2.waitKey(0)
	'''
	#----------------
	# WAY 2: OpenCV2 library
	# Apply template Matching
	res = cv2.matchTemplate(image_source,image_template, cv2.TM_CCORR_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	print("top_left", top_left)
	bottom_right = (top_left[0] + w, top_left[1] + h)
	cv2.rectangle(image_source_BGR,top_left, bottom_right, (255, 255, 0), 2)
	cv2.imshow("Conv_result", image_source_BGR)
	cv2.waitKey(0)

