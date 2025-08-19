from sub_leaf_dark import LEAF_DARK_CUTTING
from utilities import *
output = './OUTPUT/IMAGE_PRE4'
outputcut = './OUTPUT/CUT4'
leaf_dart = LEAF_DARK_CUTTING(output, outputcut)
import pickle
def find_point_dark_cutting(img,mask, angle):
    ret, thresh = cv2.threshold(mask, 127, 255,0)
    contours,hierarchy = cv2.findContours(thresh,2,1)
    cnt = max(contours, key = cv2.contourArea)
    #-----------S1 rect, rect_rotated--------------
    cnt_rotated, center_point,_ = rotate_contour(cnt, angle,None)
    x,y,w,h = cv2.boundingRect(cnt_rotated)
    rect_rotated = np.array([[[x, y]],[[x+w, y]],[[x+w, y+h]],[[x, y+h]]])
    rect_out, _,_ = rotate_contour(rect_rotated, -angle, center_point)    # rotate back
    #-----------S2 origin_rotated-------------------
    length = w
    # convex_hull ==> far_points
    hull = cv2.convexHull(cnt_rotated,returnPoints = False) # returnPoints = False while finding convex hull, in order to find convexity defects.
    try:
        defects = cv2.convexityDefects(cnt_rotated,hull)
    except:
        cnt = cv2.approxPolyDP(cnt_rotated,0.01*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        # pass
    far_points = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]     #[ start point, end point, farthest point, approximate distance to farthest point ].
        far = tuple(cnt_rotated[f][0])
        far_points.append(far)
    percent = 10
    far_points_filtered = []
    for point in far_points:
        if (x <= point[0] <= x+int(length*percent/100)) and (y <= point[1] <= y+h):
            far_points_filtered.append(point)
    y_min, y_max = 10000, 0
    for point in far_points_filtered:
        if point[1] < y_min:
            y_min = point[1]
        if point[1] > y_max:
            y_max = point[1]
    if not far_points_filtered: 
        origin_rotated = np.array([[[x, center_point[1]]]])
        print("cannot defect hull")
    else:
        origin_rotated = np.array([[[x, (y_min + y_max)//2]]])
    origin,_,diem = rotate_contour(origin_rotated, -angle, center_point)
    origin = tuple(origin.squeeze())
    # origin + angle + length ==> end_point
    x_end = int(origin[0] + length * math.cos(math.radians(angle)))
    y_end = int(origin[1] - length * math.sin(math.radians(angle)))
    end_point = (x_end, y_end)      
    # ------------------------------  
    # cv2.drawContours(img, [rect_out], 0, (255, 0, 0), 2)
    cv2.circle(img, origin,3,[0,0,255],-1)
    cv2.arrowedLine(img, origin, end_point, (255,0,0), 2)
    return origin, length, img
if __name__ == "__main__":
    # # image_path = "/home/airlab/Desktop/detectron2/demo/1_2buds/1637.jpg"
    image_path = "/home/airlab/Desktop/detectron2/demo/images_2bud_1400x1400/1000.jpg"
    im = leaf_dart.operation(image_path=image_path, separate=True)
    cv2.imshow("image", resized_img(im, 50))
    cv2.waitKey(0)
    # # MODEL 3
    # device = 'cpu'
    # folder1 = "/log/dark" #99.67
    # path1 = ''.join([current_dir, folder1])
    # path_cfg = os.path.join(path1,"dark_model.pickle")

    # with open(path_cfg, 'rb') as f:
    #     cfg2 = pickle.load(f)
    # cfg2.defrost()
    # cfg2.MODEL.WEIGHTS = os.path.join(path1,"model_final.pth")
    # cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # cfg2.MODEL.DEVICE = device
    # image_path = "/home/airlab/Desktop/detectron2/demo/images_2bud_1400x1400/1000.jpg"
    # image_ori = cv2.imread(image_path)
    # mymeta_2 = MetadataCatalog.get("orchid").set(thing_classes =["dark"])
    # predictor_2 = DefaultPredictor(cfg2)
    # out_pred_2 = predictor_2(image_ori)
    # v2 = Visualizer(image_ori[:,:,::-1], metadata = mymeta_2,scale = 0.5)
    # v2 = v2.draw_instance_predictions(out_pred_2["instances"].to("cpu"))    
    # # name_pred_1 = str(os.path.basename(image)).split(".")[0] + "_pred_single" 
    # # name_pred_2 = str(number_img) + "_pred_dark" 
    # pred_img_2 = cv2.resize(v2.get_image()[:,:,::-1],(640,456),interpolation = cv2.INTER_AREA)
    # # cv2.imshow("image", pred_img_2)
    # # cv2.waitKey(0)
    # # save_img(OUTPUT,name_pred_2,pred_img_2)
    # masks_dark = out_pred_2["instances"].pred_masks.to("cpu").detach().numpy()
    # mask_dark = np.ascontiguousarray(masks_dark)
    # mask_dark = (mask_dark*255).astype("uint8")
    # angle = [320,220]
    # for i, mask in enumerate(mask_dark):
    #     origin, length, img = find_point_dark_cutting(image_ori,mask,angle[i])
    #     cv2.imshow("image", resized_img(img,50))
    #     cv2.waitKey(0)