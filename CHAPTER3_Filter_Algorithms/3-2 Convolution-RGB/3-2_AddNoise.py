import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def main():

	img_path = "C:\\Users\\ptthi\\OneDrive\\Desktop\\PRACTICE_ON_COMPUTER_VISION\\CHAPTER3_Filter_Algorithms\\3-2 Convolution-RGB\\turtle.jpg"
	img = cv2.imread(img_path, 1) # takes img as bgr
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	rows, columns, channels = img.shape # returns 3 values which is about the given image.
	p = 0.02 # If we reduce this value, noises will become less than before on the image.
	output = np.zeros(img.shape, dtype = np.uint8)  
	for i in range(rows):
		for j in range(columns):
			r = random.random() # creates a random value is named r 
			if r < p/2:
				output[i][j] = [0, 0, 0] #pepper noise sprinkles as 0
			elif r < p:
				output[i][j] = [255, 255, 255] #salt noise sprinkles as 1
			else:
				output[i][j] = img[i][j]
    
	plt.imshow(output)
	plt.title('Image with "Salt & Pepper" Noise')
	plt.show()
	out_img = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
	cv2.imwrite("C:\\Users\\ptthi\\OneDrive\\Desktop\\PRACTICE_ON_COMPUTER_VISION\\CHAPTER3_Filter_Algorithms\\3-2 Convolution-RGB\\turtle_noise.png", out_img)


if __name__ == '__main__':
		main()


