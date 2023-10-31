import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import os.path as osp
from tqdm import tqdm
current_dir = os.getcwd()
print("path", current_dir)
# # Path 
# path = os.path.join(current_dir, name) 
# Folder Name:
name = "Images"

# Tạo folder
# if not os.path.exists(current_dir + '/' + "{}".format(name)):
os.makedirs(current_dir + '/' + '{}'.format(name),  exist_ok=True)
path_folder = current_dir + '/' + "{}".format(name)
# cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap = cv2.VideoCapture(1)
print("Test")
number_image = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    k = cv2.waitKey(1) 
    if k == ord('q'):
        break
    elif k == ord("C") or k == ord("c"):
        image_name = path_folder + '/' + '{}.png'.format(number_image)
        cv2.imwrite(image_name, gray)
        number_image += 1
        print("SAVE IMAGE")
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


