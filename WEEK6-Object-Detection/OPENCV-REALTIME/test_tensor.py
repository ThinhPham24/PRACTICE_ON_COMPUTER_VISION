# Custom deep learning developmentation 
import os
import numpy as np
import glob
from tqdm import tqdm
import cv2
import time
# import pyyaml module
import yaml
from yaml.loader import SafeLoader
from PIL import Image
import torch 

# Input folder
folder = '/Shuffle_Images_v1'
# folder = '/fruit'
current_dir = os.getcwd()
path = ''.join([current_dir, folder])
print("path = ", path)
folder_out = '/OUTPUT'
path_out = ''.join([current_dir, folder_out])
if not os.path.isdir(os.path.abspath(path_out)):
        os.mkdir(path_out)
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == '__main__':
        path_img = glob.glob(path + '/' + '*.jpg')
        # Configure of model
        with open(f'{current_dir}\\orchid_config.yaml', 'r') as f:
                data = yaml.load(f, Loader=SafeLoader)
  
        # CLASSES = yaml.load(check_yaml('coco128.yaml'))['names']
        CLASSES = data['names']
        colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        for i , image_path in enumerate(path_img):
                img = cv2.imread(image_path)
                print("shape of image", img.shape)
                trs = img.transpose((2,0,1))
                print("shape of image", trs.shape)
                mat = torch.Tensor(trs)
                print("shape of image", mat.shape)
                Batch  = 5
                
                com = torch.vstack([mat for i in range(Batch)])
                com = com.reshape(5,3,720,720)
                # new = mat.reshape(Batch,3,720,720)
                print("shape of image", com.shape)
                cv2.imshow('image', img)
                k = cv2.waitKey(0)
                if k == ord('q'):
                        break
                cv2.destroyAllWindows()


