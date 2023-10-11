import cv2
import numpy as np

h1 = [20,49,60,20,17,70,139,9]
h2 = [36,59,50,14,100,234,100,50]
h3 = [56,78,35,12,56,34,26,46]
h4 = [13,56,78,90,134,123,231,244]
h5 = [24,56,12,38,97,120,160,170]
h6 = [150,56,24,4,67,78,89,200]
h7 = [34,56,78,46,78,89,90,23]

image_matrix = np.array([h1,h2,h3,h4,h5,h6,h7])
# print("ma trix", image_matrix)

kernel = np.array([(1,0,1),(0,1,1),(0,1,1)])
print("ma trix", kernel)

KERNAL_SIZE = 3
STRIDE = 1
PADDING =  (KERNAL_SIZE - STRIDE)/2
PADDING = int(PADDING)
new_matrix = np.pad(image_matrix,((PADDING,PADDING),(PADDING,PADDING)),'constant')
# print("new matrix", new_matrix)
 # Flip the kernel
kernel = np.flipud(np.fliplr(kernel))
print("matrix:", kernel)
# # convolution output
output = np.zeros_like(image_matrix)

# # # Add zero padding to the input image
# # image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
# # image_padded[1:-1, 1:-1] = image

# # Loop over every pixel of the image
for x in range((new_matrix.shape[1] - KERNAL_SIZE)+1):
	for y in range((new_matrix.shape[0] - KERNAL_SIZE) +1):
		# element-wise multiplication of the kernel and the image
		a = new_matrix[y: y+3, x: x+3]
		print("shape", a.shape)
		output[y, x] = (kernel * new_matrix[y: y+3, x: x+3]).sum()

print('output',output)

