import os
import json
from tqdm import tqdm
import shutil
import contextlib
import json

import cv2
import pandas as pd
from PIL import Image
from collections import defaultdict
import numpy as np
def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance. 
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all 
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...], 
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def delete_dsstore(path='../datasets'):
    # Delete apple .DS_store files
    from pathlib import Path
    files = list(Path(path).rglob('.DS_store'))
    print(files)
    for f in files:
        f.unlink()
def make_folders(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path
def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format: 
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format: 
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """
    
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]

def convert_coco_json_to_yolo_txt(output_path, json_file,use_segments=True):

    import pandas as pd
    import numpy as np
    import os
    make_folders(output_path)

    df_img_id = []
    df_img_name = []
    df_img_width = []
    df_img_height = []
    with open(json_file) as f:
        json_data = json.load(f)
    print('ekstraksi dari json :',json_file)

    # write _darknet.labels, which holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "_darknet.labels.txt")
    with open(label_file, "w") as f:
        for image in tqdm(json_data["images"], desc="Annotation txt for each iamge"):
            img_id = image["id"]
            #img_name = image["file_name"]
            img_name = os.path.basename(image["file_name"])
            print("checking :", img_name,"and copying images to output path")
            json_images = os.path.dirname(json_file)+"/"+image["file_name"]
            
            #cp -R {json_images} {output_path}
            img_width = image["width"]
            img_height = image["height"]
            df_img_id.append(img_id)
            df_img_name.append(img_name)
            df_img_width.append(img_width)
            df_img_height.append(img_height)
        
            anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
            #anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
            anno_txt = os.path.join(output_path, img_name.replace(".jpg",".txt"))

            h, w, f = image['height'], image['width'], image['file_name']
            bboxes = []
            segments = []
            with open(anno_txt, "w") as f:
                for anno in anno_in_image:
                    category = anno["category_id"]
                    print("Category", category)
                    print()
                    bbox_COCO = anno["bbox"]
                    #x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                    if anno['iscrowd']:
                        continue
                    # The COCO box format is [top left x, top left y, width, height]
                    box = np.array(anno['bbox'], dtype=np.float64)
                    box[:2] += box[2:] / 2  # xy top-left corner to center
                    box[[0, 2]] /= w  # normalize x
                    box[[1, 3]] /= h  # normalize y
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue
                    #cls = coco80[anno['category_id'] - 1] if cls91to80 else anno['category_id'] - 1  # class
                    cls = anno['category_id'] - 1
                    box = [cls] + box.tolist()
                    if box not in bboxes:
                        bboxes.append(box)
                    # Segments
                    if use_segments:
                        if len(anno['segmentation']) > 1:
                            s = merge_multi_segment(anno['segmentation'])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in anno['segmentation'] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        if s not in segments:
                            segments.append(s)

                    last_iter=len(bboxes)-1
                    line = *(segments[last_iter] if use_segments else bboxes[last_iter]),  # cls, box or segments
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                print("that images contains class:",len(bboxes),"objects")

    #print("Converting COCO Json to YOLO txt finished!")
    #json_images = os.path.dirname(json_file)+"/*.jpg"
    
    #print("Copying images to output path")
    #!cp -R {json_images} {output_path}
    
    print("creating category_id and category name in darknet.labels")
    with open(label_file, "w") as f:
        for category in tqdm(json_data["categories"], desc="Categories"):
            category_name = category["name"]
            print(category_name)
            f.write(f"{category_name}\n")

    print("Finish")