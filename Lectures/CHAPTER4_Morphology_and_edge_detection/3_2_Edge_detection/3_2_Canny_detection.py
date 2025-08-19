import cv2
import os
import numpy as np

#-------------------------------------------------------
def canny_edge_detection(image, sigma=1, kernel_size=3, low_threshold=10, high_threshold=150):
    """Canny edge detection algorithm implementation from scratch"""
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian filter to smooth the image
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    # Calculate the gradient magnitude and direction using Sobel operators
    dx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(dx*dx + dy*dy)
    angle = np.arctan2(dy, dx) * 180 / np.pi
    # Quantize the gradient direction into 4 possible angles: 0, 45, 90, and 135 degrees
    angle[angle < 0] += 180
    angle[angle >= 157.5] = 0
    angle[(angle >= 22.5) & (angle < 67.5)] = 45
    angle[(angle >= 67.5) & (angle < 112.5)] = 90
    angle[(angle >= 112.5) & (angle < 157.5)] = 135
    # Apply non-maximum suppression to thin the edges
    suppressed = np.zeros_like(mag)
    for i in range(1, mag.shape[0]-1):
        for j in range(1, mag.shape[1]-1):
            direction = angle[i, j]
            if direction == 0:
                if mag[i, j] > mag[i, j-1] and mag[i, j] > mag[i, j+1]:
                    suppressed[i, j] = mag[i, j]
            elif direction == 45:
                if mag[i, j] > mag[i-1, j+1] and mag[i, j] > mag[i+1, j-1]:
                    suppressed[i, j] = mag[i, j]
            elif direction == 90:
                if mag[i, j] > mag[i-1, j] and mag[i, j] > mag[i+1, j]:
                    suppressed[i, j] = mag[i, j]
            elif direction == 135:
                if mag[i, j] > mag[i-1, j-1] and mag[i, j] > mag[i+1, j+1]:
                    suppressed[i, j] = mag[i, j]
    # Apply hysteresis thresholding to determine the final edges
    strong_edges = suppressed > high_threshold
    weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)
    edges = np.zeros_like(suppressed)
    edges[strong_edges] = 255
    for i in range(1, edges.shape[0]-1):
        for j in range(1, edges.shape[1]-1):
            if weak_edges[i, j]:
                if (edges[i-1:i+2, j-1:j+2] > high_threshold).any():
                    edges[i, j] = 255
    return edges
#----------------------------------------

#defining the custom canny detector
def Canny_detector(img,weak_th=None,strong_th=None):
    #conversion of image to grayscale
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Noise reduction step
    # img=cv2.GaussianBlur(img,(5,5),1.4)
    #Calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0,3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1,3)
    mag, ang = cv2.cartToPolar(gx, gy,angleInDegrees=True)
    #setting the minimum and maximum thresholds for double thresholding
    mag_max=np.max(mag)
    if not weak_th:weak_th=mag_max*0.1
    if not strong_th:strong_th=mag_max*0.5   
    height,width=img.shape
    for i_x in range(width):
        for i_y in range(height):    
            grad_ang=ang[i_y,i_x]
            grad_ang=abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)
            #selecting the neigbours of the target pixel according to the gradient direction
            if grad_ang<=22.5:
                neighb_1_x,neighb_1_y=i_x-1,i_y
                neighb_2_x,neighb_2_y=i_x+1,i_y
            elif grad_ang>22.5 and grad_ang<=(22.5+45):
                neighb_1_x,neighb_1_y=i_x-1,i_y-1
                neighb_2_x,neighb_2_y=i_x+1,i_y+1
            elif grad_ang>(22.5+45) and grad_ang<=(22.5+90):
                neighb_1_x,neighb_1_y=i_x,i_y-1
                neighb_2_x,neighb_2_y=i_x,i_y+1
            elif grad_ang>(22.5+90) and grad_ang<=(22.5+135):
                neighb_1_x,neighb_1_y=i_x-1,i_y+1
                neighb_2_x,neighb_2_y=i_x+1,i_y-1
            elif grad_ang>(22.5+135) and grad_ang<=(22.5+180):
                neighb_1_x,neighb_1_y=i_x-1,i_y
                neighb_2_x,neighb_2_y=i_x+1,i_y
            #Non-maximum supression step
            if width>neighb_1_x>=0 and height>neighb_1_y>=0:
                if mag[i_y,i_x]<mag[neighb_1_y,neighb_1_x]:
                    mag[i_y,i_x]=0
                    continue
            if width>neighb_2_x>=0 and height>neighb_2_y>=0:
                if mag[i_y,i_x]<mag[neighb_2_y,neighb_2_x]:
                    mag[i_y,i_x]=0
    weak_ids= np.zeros_like(img)
    strong_ids= np.zeros_like(img)              
    ids=np.zeros_like(img)
    #double thresholding step
    for i_x in range(width):
        for i_y in range(height):
            grad_mag=mag[i_y,i_x]
            if grad_mag<weak_th:
                mag[i_y,i_x]=0
            elif strong_th>grad_mag>=weak_th:
                ids[i_y,i_x]=1
            else:
                ids[i_y,i_x]=2
    return ids[1:5,1:5]

if __name__ == "__main__":
    # folder = "\images"
    # current_dir = os.getcwd()
    # print("current_dir",current_dir)
    # path = ''.join([current_dir, folder])
    filepath = "C:\\Users\\ptthi\\OneDrive\\Desktop\\PRACTICE_ON_COMPUTER_VISION\\CHAPTER4_Morphology_and_edge_detection\\3_2_Edge_detection\\turtle.jpg"
    image = cv2.imread(filepath)
    # edges = canny_edge_detection(image,sigma=1.4,kernel_size=5,low_threshold=10,high_threshold=150)
    # aperture_size = 3 # Aperture size
    # L2Gradient = False  # Boolean
    # # Applying the Canny Edge filter 
    # # with Aperture Size and L2Gradient
    # edges = cv2.Canny(image, 10, 150,
    #              apertureSize = aperture_size, 
    #              L2gradient = L2Gradient )
    image_array = np.array([[20,20,30,40,40,40],[20,22,30,37,40,40],[30,30,30,30,30,30],[40,37,30,25,22,22],[40,40,30,25,25,25],[40,40,30,25,27,20]],dtype=np.int8)
    # print(image_array)
    edges = Canny_detector(image_array,40,255) 
    # edges = cv2.Canny(image_array, 10, 150)
    print(edges)
    # cv2.imshow("image",image)
    # cv2.imshow("edge image", edges)
    # cv2.waitKey(0)
