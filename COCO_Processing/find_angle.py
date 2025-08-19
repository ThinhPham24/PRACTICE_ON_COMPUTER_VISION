from glob import glob
import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import math 
folder = "/DATA_TEST_AUGMENTATION"
mask_dir = "ANNOTATION/train"
img_dir = "IMG/train"
current_dir = os.getcwd()
path= ''.join([current_dir, folder])
mask_path = os.path.join(path, mask_dir)
img_path = os.path.join(path, img_dir)
print(mask_path)
if __name__ =="__main__":
    filename_mask = sorted(glob.glob(mask_path + "/*.png"))
    filename_img = sorted(glob.glob(img_path + "/*.jpg"))
    ind_num =0
    for n, image_file in tqdm(enumerate(filename_img),total=len(filename_img)):
        # print("file of image", image_file)
        annotation_files =[]
        ind_image = str(os.path.basename(image_file)).split(".")[0]
        image = cv2.imread(image_file)
        for i,file in enumerate(filename_mask):
            # print("len of flie", len(file))
            if ind_num < (len(filename_mask)):
                ind_anno = str(os.path.basename(filename_mask[ind_num])).split(".")[0].split('_bud')[0]
                # print("ind_anno,ind_image",(ind_anno,ind_image))
                # print("id number", ind_num)
            if ind_num < (len(filename_mask)) and ind_anno == ind_image:
                # print("file annotation", os.path.basename(filename_mask[ind_num]))
                # print("file names", len(filenames))
                # print("file names", len(filenames_an))
                # print("so luong",ind_num )
                annotation_files.append(filename_mask[ind_num])
                ind_num +=1
            else: 
                if None in annotation_files :
                    continue
                else:
                    points =[]
                    # print("all annotation_files,",annotation_files)
                    for annotation_filename in annotation_files:
                        # print("file name:",annotation_filename)
                        # [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
                        # x_axis      = np.array([1, 0])    # unit vector in the same direction as the x axis
                        # your_line   = np.array([vx, vy])  # unit vector in the same direction as your line
                        # dot_product = np.dot(x_axis, your_line)
                        # angle_2_x   = np.arccos(dot_product)
                        mask = cv2.imread(annotation_filename)
                        gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(gray, 127, 255,0)
                        contours,hierarchy = cv2.findContours(thresh,2,1)
                        cnt = max(contours, key = cv2.contourArea)
                        vx, vy, cx, cy = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
                        # cv2.DIST_HUBER
                        x, y, w, h = cv2.boundingRect(cnt)
                        hull = cv2.convexHull(cnt,False)
                        minRect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(minRect)
                        box = np.int0(box)
                        moment = cv2.moments(cnt)
                        cen_X = int(moment['m10'] / moment['m00'])
                        cen_Y = int(moment['m01'] / moment['m00'])



                        cv2.drawContours(image,[box], -1, (170, 255, 175), 3)
                        cv2.line(image, (int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)), (0, 0, 255))
                        cv2.rectangle(image,(x,y),(x+w,y+h),(100,100,100),2 )
                        cv2.drawContours(image,cnt, -1, (255, 255, 0), 3)
                        cv2.drawContours(image,[hull], -1, (255, 0, 100), 1, 8)
                        line_angle = math.atan2(vy, vx)
                        line_angle_degrees = math.degrees(line_angle)
                        angle = line_angle_degrees 
                        # cv2.putText(image,f"{round(angle,0)}", (int(cx),int(cy)),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
                        cv2.circle(image,(int(cx),int(cy)),2,(0,0,255),2)
                        cv2.circle(image,(int(cen_X),int(cen_Y)),2,(0,0,0),2)
                    cv2.imshow("image", image)
                    k= cv2.waitKey(0)
                    if k == ord('b'):
                        break
                    break
