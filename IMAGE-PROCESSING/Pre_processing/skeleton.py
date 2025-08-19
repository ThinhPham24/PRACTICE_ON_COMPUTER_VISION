import numpy as np
import cv2
from plantcv import plantcv as pcv
from matplotlib import pyplot as plt                                                              
if __name__ == '__main__':
    path_img = "C:\\Users\\ptthi\\OneDrive\\Desktop\\Image_processing\\THINH\\276.jpg"
    color_image = cv2.imread(path_img)
    gray_image = cv2.cvtColor(color_image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image,(5,5),0)
    _, thresh = cv2.threshold(blur,220,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    shape = cv2.MORPH_ELLIPSE
    ksize = (7, 7)
    kernel = cv2.getStructuringElement(shape,ksize)
    eroded_image = cv2.erode(thresh, kernel)
    dilated_image = cv2.dilate(eroded_image, kernel)
    skeleton = pcv.morphology.skeletonize(mask=dilated_image) 
    point = np.where(skeleton==255)
    x_ske = []
    y_ske = []
    for y, x in zip(point[0], point[1]):
        print("White pixel found at coordinates (x={}, y={})".format(x, y))
        x_ske.append(x)
        y_ske.append(y)
        cv2.circle(color_image,(x,y),radius=1,color=(255,0,0),thickness = 1)
    # print("ALL POINT OF SKELETON", point)
    cv2.imshow("image", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()