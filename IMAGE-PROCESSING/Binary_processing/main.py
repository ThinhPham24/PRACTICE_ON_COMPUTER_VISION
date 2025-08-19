import cv2
import os
import glob
import re
import fnmatch
import numpy as np

folder = "/IMG/"
current_dir = os.getcwd()
path = ''.join([current_dir, folder])

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files
def processing_image(image,set_area = 1500):

    image_test = image[20:image.shape[0] - 20, 20:image.shape[1] - 20]
    blank_img = np.zeros([image_test.shape[0], image_test.shape[1], 3], dtype=np.uint8)
    image_gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(image_gray, (5, 5))
    ret, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
    erode = cv2.erode(thresh, (3, 3), 0)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel_di = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(opening, (3, 3), iterations=2)  # 11
    kernel = np.ones((11, 11), np.uint8)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=3)
    kernel = np.ones((7, 7), np.uint8)
    erode2 = cv2.erode(closing, kernel, iterations=2)
    kernel = np.ones((5, 5), np.uint8)
    dilate2 = cv2.dilate(erode2, kernel, iterations=4)
    contours, _ = cv2.findContours(dilate2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_add = []
    if len(contours) is not None:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > set_area:
                cnt_add.append(cnt)
    for cnt2 in cnt_add:
        cv2.fillPoly(blank_img, pts=[cnt2], color=(255, 255, 255))
        cv2.drawContours(image_test, [cnt2], -1, (0, 255, 0), 3)
    return blank_img, image_test
if __name__ == "__main__":
    for root,_, files in os.walk(path):
        for file in files:
            image_path = os.path.join(root,file)
            print("image",image_path)
            image = cv2.imread(image_path)
            image_binary, image_with_contour = processing_image(image,set_area=2500)
            cv2.imshow("image_binary", image_binary)
            cv2.imshow("image_with_contour", image_with_contour)
            cv2.imwrite(f"image_binary_{file}.png",image_binary)
            cv2.waitKey(0)


