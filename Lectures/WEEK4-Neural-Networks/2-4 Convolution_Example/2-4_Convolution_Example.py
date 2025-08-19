import cv2
import os
import numpy as np
from torch import max_pool2d
import torch


KERNAL_SIZE = 3
STRIDE = 1
PADDING =  (KERNAL_SIZE - STRIDE)/2
PADDING = int(PADDING)
img = cv2.imread('./006_01_01_051_08.png')
img = cv2.resize(img,(28,32))
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
Conv_Filter = np.random.rand(KERNAL_SIZE,KERNAL_SIZE)
# print(Conv_Filter)
#Normal distribution
# mean = 2
# std = 0
#Conv_Filter = np.random.normal(mean, std, (KERNAL_SIZE,KERNAL_SIZE))
Conv_Filter = Conv_Filter/np.sum(Conv_Filter)
#print(Conv_Filter)
img_F = img
def pooling(inputMap,poolSize=5,poolStride=1,mode=''):
    # inputMap sizes
    in_row,in_col = np.shape(inputMap)
    # outputMap sizes
    out_row,out_col = int(np.floor(in_row/poolStride)),int(np.floor(in_col/poolStride))
    row_remainder,col_remainder = np.mod(in_row,poolStride),np.mod(in_col,poolStride)
    if row_remainder != 0:
        out_row +=1
    if col_remainder != 0:
        out_col +=1
    outputMap = np.zeros((out_row,out_col))
    # padding
    temp_map = np.lib.pad(inputMap, ((0,poolSize-row_remainder),(0,poolSize-col_remainder)), 'edge')
    
    # max pooling
    if mode == 'max':
        for r_idx in range(0,out_row):
            for c_idx in range(0,out_col):
                startX = c_idx * poolStride
                startY = r_idx * poolStride
                poolField = temp_map[startY:startY + poolSize, startX:startX + poolSize]
                poolOut = np.max(poolField)
                outputMap[r_idx,c_idx] = poolOut
        
        # retrun outputMap
        return  outputMap
    if mode == 'average':
        for r_idx in range(0,out_row):
            for c_idx in range(0,out_col):
                startX = c_idx * poolStride
                startY = r_idx * poolStride
                poolField = temp_map[startY:startY + poolSize, startX:startX + poolSize]
                poolOut = np.mean(poolField)
                outputMap[r_idx,c_idx] = poolOut
        
        # retrun outputMap
        return  outputMap

cv2.waitKey(0)
if (PADDING % 1) == 0:
    img_F = np.pad(img_F,((PADDING,PADDING),(PADDING,PADDING)),'constant')
else:
    img_F = np.pad(img_F,((PADDING*2,PADDING*2),(0,0)),'constant')
[H,W]=np.shape(img_F)

new_feature = np.zeros((int((H-KERNAL_SIZE)/STRIDE)+1,int((W-KERNAL_SIZE)/STRIDE)+1))

for h in range(int((H-KERNAL_SIZE)/STRIDE)+1):
    for w in range(int((W-KERNAL_SIZE)/STRIDE)+1):
        aa = (img_F[h*STRIDE:h*STRIDE + (KERNAL_SIZE), w*STRIDE:w*STRIDE + (KERNAL_SIZE)]*Conv_Filter)
        #print(aa)
        new_feature[h,w] = np.sum(aa)

        img_S = img_F.astype(np.uint8)
        img_new = new_feature.astype(np.uint8)

        cv2.rectangle(img_S, (int(w*STRIDE), int(h*STRIDE)), (int((w*STRIDE + KERNAL_SIZE)), int((h*STRIDE + KERNAL_SIZE))), (255, 0, 0), 1)
        # cv2.rectangle(img_S, (int(w+STRIDE-1), int(h+STRIDE-1)), (int((w+STRIDE-1 + KERNAL_SIZE)), int((h+STRIDE-1 + KERNAL_SIZE))), (255, 0, 0), 1)
        cv2.namedWindow('Conv_process', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Conv_process", 300, 300)
        cv2.imshow('Conv_process',img_S)

        cv2.namedWindow('Conv_result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Conv_result", 300, 300)
        cv2.imshow('Conv_result',img_new)

        cv2.waitKey(1)

#img_new = pooling(img_new, 3, 1, 'average')
img_new = img_new.astype(np.uint8)

Conv_Img = img_new

cv2.namedWindow('Convresult', cv2.WINDOW_NORMAL)
cv2.resizeWindow("Convresult", 300, 300)
cv2.imshow('Convresult',Conv_Img)
cv2.waitKey(0)

imgd = np.reshape(img, np.size(img))
convimgd = np.reshape(Conv_Img, np.size(Conv_Img))


print(convimgd.shape)
print(imgd.shape)

Diff = np.sum(abs((imgd)/ np.linalg.norm((imgd)) - (convimgd)/ np.linalg.norm(convimgd)))

PercnetageDiff = Diff / np.sum((imgd)/ np.linalg.norm(imgd))*100

print('The information loss using %dx%d convolution kernel is %.6f%%\n\n' %(KERNAL_SIZE,KERNAL_SIZE,PercnetageDiff))


