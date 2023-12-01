# Custom deep learning developmentation 
import os
import numpy as np
import cv2
# import pyyaml module
import yaml
from yaml.loader import SafeLoader

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h,CLASSES):
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
class Net_detection:
        def __init__(self, path:str, file:str, cls:list):
                super(Net_detection, self).__init__()
                self.CLASSES = cls
                self.model_path = path
                self.config_file = file
                self.model = cv2.dnn.readNet(model = self.model_path, config = self.config_file, framework='onnx')
                # net = cv2.dnn.readNetFromONNX(f'{current_dir}\\best_yolo_seg.onnx')
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                # model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                # model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        def apply(self, image_path, image_size = (736, 736)):
                orgin_img = cv2.imread(image_path)
                img =  orgin_img.copy()
                [height, width, _] = img.shape
                length = max((height, width))
                scale_x = width/image_size[1]  #weight
                scale_y = height/image_size[0] #height
                blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size= image_size, swapRB=True)
                # set the input blob for the neural network
                self.model.setInput(blob)
                # forward pass image blog through the model
                outputs = self.model.forward("output0") #  (B,A,W,H )(B,anchor,W,H)  
                # outputs = self.model.forward("output1") #  (B,32,W,H )(Batch size, depth map, weight, height) 
                outputs = np.array([cv2.transpose(outputs[0])])
                rows = outputs.shape[1]
                boxes = []
                scores = []
                class_ids = []
                masks= []
                for i in range(rows):
                        classes_scores = outputs[0][i][4:6]
                        # all_masks = outputs[0][i][6:]
                        (minScore, maxScore, minClassLoc,(x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                        if maxScore >= 0.25:
                                box = [outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                                        outputs[0][i][2], outputs[0][i][3]] # Becuase we want a format as (x1,y1,W,H) (center_x - W,center_y -H,W,H)
                                boxes.append(box)
                                scores.append(maxScore)
                                class_ids.append(maxClassIndex)
                                mask = outputs[0][i][6:]
                                masks.append(mask)
                result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
                detections = []
                for i in range(len(result_boxes)):
                        index = result_boxes[i]
                        box = boxes[index]
                        detection = {
                        'class_id': class_ids[index],
                        'class_name': self.CLASSES[class_ids[index]],
                        'confidence': scores[index],
                        'box': box}
                        detections.append(detection)
                        draw_bounding_box(img, class_ids[index], scores[index], round(box[0] * scale_x), round(box[1] * scale_y),
                                        round((box[0] + box[2]) * scale_x), round((box[1] + box[3]) * scale_y),self.CLASSES)
                        mask_nms = masks[index]
                re_image_pred = cv2.resize(img,(500,500),interpolation=cv2.INTER_AREA)
                re_origin_img = cv2.resize(orgin_img,(500,500),interpolation=cv2.INTER_AREA)
                merge_img = np.hstack([re_origin_img,re_image_pred])
                return merge_img
                

