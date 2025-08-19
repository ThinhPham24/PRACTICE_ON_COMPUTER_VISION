import cv2 as cv
import numpy as np
from scipy.spatial import distance
import os
from tqdm import tqdm

def insert_text_before_extension(text, substring, insert_text):
    if substring in text:
        modified_text = text.replace(".png", insert_text + ".png")
        return modified_text
    else:
        return None
def remove_letter_from_filename(filename, letter):
    parts = filename.split(letter)
    new_filename = ''.join(parts)
    return new_filename
def scale(src,scale):
    #calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale / 100)
    height = int(src.shape[0] * scale / 100)
    # dsize
    dsize = (width, height)
    # resize image
    return cv.resize(src, dsize)

def fill_bg(image):
    # convert image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # apply thresholding to binarize image
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # find contours in the binary image
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # get the contour with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    
    # create a mask with the same size as the input image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # draw the largest contour on the mask with white color
    if max_contour is not None:
        cv.drawContours(mask, [max_contour], -1, 255, cv.FILLED)

    # apply the mask to the input image to fill the background with white
    result = cv.bitwise_and(image, image, mask=mask)
    result[mask == 0] = 255
    result=cv.GaussianBlur(result, (3,3), 0)
    # return the image with the background filled with white
    return result

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

'Input'
w=720
h=720
input_images= '/IMAGE_2022_09_05' 
output_images = '/CropImage/Sub_Image_21' 

cwd = os.getcwd()
img_path= cwd + input_images
out_path= cwd + output_images


if not os.path.exists(cwd+output_images):
    os.makedirs(cwd+output_images)

# print(os.listdir(img_path))
# input("enter")
for image_path in tqdm(os.listdir(img_path)):
    input_image = os.path.join(img_path, image_path)
    print(input_image)
    input("enter")
    try:
        img = cv.imread(input_image)
        img = img[100:100+1948, 100:100+2092]
        # cv.imshow('text',scale(img,50))
        # cv.waitKey(0)
        img = scale(img,75)

        copy=img.copy()


        # img=fill_bg(img)

        original = img
        image_contours = np.zeros((img.shape[1], img.shape[0], 1), np.uint8)
        image_binary = np.zeros((img.shape[1], img.shape[0], 1), np.uint8)

        for channel in range(img.shape[2]):
            ret, image_thresh = cv.threshold(img[:, :, channel], 100, 255, cv.THRESH_OTSU)    
            contours = cv.findContours(image_thresh, 1, 1)[0]   
            cv.drawContours(image_contours, contours, -1, (255,255,255), 3)

        contours = cv.findContours(image_contours, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[0]
        cv.drawContours(image_binary, [max(contours, key = cv.contourArea)],-1, (255, 255, 255), -1)

        contours = cv.findContours(image_binary, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[0]
        # cv.imshow('text',scale(image_binary,50))
        # cv.waitKey(0)


        moment = cv.moments(image_binary) 
        X = int(moment ["m10"] / moment["m00"]) 
        Y = int(moment ["m01"] / moment["m00"]) 
        center = (X,Y)

        # cv.circle(original, center, 3, (255, 100, 0), 2)

        cnt=[]
        for contour in contours:
            M = cv.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                contour_center=(cx,cy)

                # calculate distance to image_center
                distances_to_center = (distance.euclidean(center, contour_center))
        
                # print("distances_to_center:" , distances_to_center )
                # save to a list of dictionaries
                cnt.append({'contour': contour, 'center': contour_center, 'distance_to_center': distances_to_center})
                
                # draw each contour (red)
                cv.drawContours(original, [contour], 0, (0, 50, 255), 2)

                cv.circle(original, contour_center, 10, (0,0,255), -1)

                # cv.imshow('text',scale(original,50))
                # cv.waitKey(0)



        sorted_cnt = sorted(cnt, key=lambda i: i['distance_to_center'])

        #  find contour of closest building to center and draw it (blue)
        center_main_contour = sorted_cnt[0]['contour']
        # cv.drawContours(original, [center_main_contour], 0, (255, 0, 0), 2)


        (ccx,ccy),radius = cv.minEnclosingCircle(center_main_contour)



        crop=copy[int(ccy-w/2):int(ccy+w/2), int(ccx-h/2):int(ccx+h/2)]

        extension = os.path.splitext(image_path)[1]
        fullname=os.path.splitext(image_path)[0]  + extension
        if 'Ltop' in fullname:
            nameInsert = insert_text_before_extension(fullname, 'Ltop', '_4')
            name =  remove_letter_from_filename(nameInsert, 'Ltop_')
        elif 'L0' in fullname:
            nameInsert = insert_text_before_extension(fullname, 'L0', '_1')
            name =  remove_letter_from_filename(nameInsert, 'L0_')
        elif 'L120' in fullname:
            nameInsert = insert_text_before_extension(fullname, 'L120', '_2')

            name =  remove_letter_from_filename(nameInsert, 'L120_')
        elif 'L240' in fullname:
            nameInsert = insert_text_before_extension(fullname, 'L240', '_3')

            name =  remove_letter_from_filename(nameInsert, 'L240_')
        else: 
            name =  remove_letter_from_filename(fullname, '_L')
        fullpath = os.path.join(out_path,name)

        if not os.path.exists(fullpath):
            print('create')
            cv.imwrite(fullpath, crop)
    except:
        print("error")
        input("enter")
        continue
        

    