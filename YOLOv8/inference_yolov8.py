# Ultralytics YOLO 🚀, AGPL-3.0 license
from ultralytics import YOLO
from ultralytics.engine.results import Results
import os
import numpy as np
import glob
from tqdm import tqdm
import cv2
import time
# Input folder
folder = '\Shuffle_Images_v1'
# folder = '/fruit'
current_dir = os.getcwd()
path = ''.join([current_dir, folder])
print("path = ", path)
# Folder output
folder_out = '\OUTPUT'
path_out = ''.join([current_dir, folder_out])
if not os.path.isdir(os.path.abspath(path_out)):
        os.mkdir(path_out)

if __name__ == '__main__':
    # # Load the model.
    p=current_dir+'\\best_yolov8.pt'
    print("p",p)
    model = YOLO(p,task = 'segment') 
    threshold = 0.5
    path_img = glob.glob(path + '/' + '*.jpg')
    t1 = time.time()
    for i, image in tqdm(enumerate(path_img),total=len(path_img), desc = "PROCESSING"):
        print("image name:",image)
        base_name = os.path.basename(image) 
        # imagesz = cv2.imread(image)
        # cv2.imshow("img",r)
        r = model.predict(source = f'{image}',stream=False,conf = threshold)[0] #Set confident score = 0.8
        image_pred = Results(orig_img=r.orig_img,path=r.path,names= r.names,boxes=r.boxes.data,masks=r.masks.data).plot()
        out_image = os.path.join(path_out,base_name)
        cv2.imwrite(out_image,image_pred)
        cv2.imshow("predicted image", image_pred)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
            
    print("Time Total:", (time.time()-t1))

