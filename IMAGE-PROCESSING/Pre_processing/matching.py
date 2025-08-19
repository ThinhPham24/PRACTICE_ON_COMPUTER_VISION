import numpy as np
def cal_center(A):
    if len(A) >= 1:
        center= []
        for i, box in enumerate(A):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center.append((center_x,center_y))
        return center
    else:
        print('greater than 1')
        center = None
        return center
def calculate_iou(boxA, boxB):
    # Extract coordinates from the bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # Calculate the area of intersection rectangle
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # Calculate the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # Calculate the IoU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou
def match_boxes(boxes1, boxes2, threshold):
    matched_pairs = []
    unmatched_boxes2 = list(range(len(boxes2)))
    for i, box1 in enumerate(boxes1):
        best_iou = 0.0
        best_match_index = -1
        for j, box2 in enumerate(boxes2):
            iou = calculate_iou(box1, box2)
            if iou >= threshold and iou > best_iou:
                best_iou = iou
                best_match_index = j
        if best_match_index != -1:
            matched_pairs.append((i, best_match_index))
            unmatched_boxes2.remove(best_match_index)
    unmatched_boxes1 = [i for i in range(len(boxes1)) if i not in [pair[0] for pair in matched_pairs]]
    return matched_pairs, unmatched_boxes1, unmatched_boxes2
if __name__ == "__main__":
    #Coordination of each bounding box (x1,y1,x2,y2)
    box_1 = [(240,80,320,390),(120,240,280,420),(60,280,260,460)]
    box_2 = [(130,245,280,435),(65,275,275,445), (0,0,200,200)] #(235,85,310,385)
    matched_pairs, unmatched_boxes1, unmatched_boxes2 = match_boxes(boxes1=box_1,boxes2=box_2,threshold=0.5)
    print("matched pairs", matched_pairs)
    print("unmatched pairs 1", unmatched_boxes1)
    print("unmatched pairs 2", unmatched_boxes2)

