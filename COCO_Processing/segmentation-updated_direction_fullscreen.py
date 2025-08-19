#NEW VERSION: BOTH WORK FOR PYTHON 3.5 AND 3.6 (OCT-24th, 2019)

#****************************************************************************************************************#
# USEAGE
# 1. python3   segmentation-updated_direction_fullscreen.py    B 	- # For annotating the each single bud
# 2. python3   segmentation-updated_direction_fullscreen.py    L 	- # For annotating the darken part
# 3. python3   segmentation-updated_direction_fullscreen.py    C 	- # For annotating the root R
# 4. python3   segmentation-updated_direction_fullscreen.py    any	- # For annotating the full bud
#****************************************************************************************************************#

import os
import numpy
import matplotlib.pyplot as plt
import cv2
import os, os.path
import glob, os
from matplotlib.widgets import RectangleSelector # tam thoi de day, segmentation khong can
from matplotlib.widgets import TextBox
import argparse
import math
#-----------------------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------#
from matplotlib import pyplot as plt
#from shapely.geometry.polygon import Polygon
#from descartes import PolygonPatch
from PIL import Image, ImageDraw
#-----------------------------------------------------------------------------------------------------#
BLUE = '#6699cc'
GRAY = '#999999'
BLACK = '#000000'
WHITE = '#FFFFFF'
my_dpi = 10
no_images = 0
global width
global height

# global constants
img = None
tl_list = [] #top-left list (luu vi tri chuot)
br_list = [] #bottom-right list (luu vi tri chuot)
object_list = []
figure = []

folder = "/datas2/"
annotations_dir = "ANNOTATION"

current_dir = os.getcwd()
path = ''.join([current_dir, folder])

annotations_savepath = os.path.join(path, annotations_dir)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.mkdir(annotations_savepath)


class LineBuilder:
    def __init__(self, line,ax,color):
        self.line = line
        self.ax = ax
        self.color = color
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.counter = 0
        self.shape_counter = 0
        self.shape = {}
        #self.precision = 10 # Original value: 10 --> reduce the pixel precision to close the shape (5)
        self.precision = 5 # Original value: 10 --> reduce the pixel precision to close the shape (5)

    def __call__(self, event):
        if event.inaxes!=self.line.axes: return
        if self.counter == 0:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
        if np.abs(event.xdata-self.xs[0])<=self.precision and np.abs(event.ydata-self.ys[0])<=self.precision and self.counter != 0:
            self.xs.append(self.xs[0])
            self.ys.append(self.ys[0])
            self.ax.scatter(self.xs,self.ys,s=10,color='yellow')  # Original value: 120
            self.ax.scatter(self.xs[0],self.ys[0],s=20,color='blue') # Original value: 80
            self.ax.plot(self.xs,self.ys,color=self.color)
            self.line.figure.canvas.draw()
            self.shape[self.shape_counter] = [self.xs,self.ys]
            self.shape_counter = self.shape_counter + 1
            self.xs = []
            self.ys = []
            self.counter = 0
        else:
            if self.counter != 0:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
            self.ax.scatter(self.xs,self.ys,s=10,color=self.color)
            self.ax.plot(self.xs,self.ys,color=self.color)
            self.line.figure.canvas.draw()
            self.counter = self.counter + 1

def create_shape_on_image(data,cmap='jet'):
    global width
    global height
    width = data[:,:,0].shape[1]
    height = data[:,:,0].shape[0]
	# This function to make sure that the shape only be created by these point that we want!
    def change_shapes(shapes):
        new_shapes = {}
        original_shape = {}
        for i in range(len(shapes)):
            l = len(shapes[i][1])
            print("tong chieu dai = ", l)
            new_shapes[i] = np.zeros((l-2, 2),dtype='int')
            original_shape[i] = np.zeros((l-2, 2), dtype='int')
            for j in range(l-2): # Minus 2 because we don't want to use the first & the last point
                new_shapes[i][j,0] = shapes[i][0][j+1] # the index of new_shapes always start at zero
                new_shapes[i][j,1] = shapes[i][1][j+1] # we need to adjust the index of shapes

                original_shape[i][j,0] = shapes[i][0][j] # For angle computation
                original_shape[i][j,1] = shapes[i][1][j] # For angle computation
        return new_shapes, original_shape

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click to include shape markers (5 pixel precision to close the shape)')
    line = ax.imshow(data) 
    ax.set_xlim(0,data[:,:,0].shape[1])
    ax.set_ylim(0,data[:,:,0].shape[0])
    # ax_box = plt.axes([0.1,0.05,0.8,0.07])
    # textbox  = TextBox(ax_box,'Name', initial = 'None' )
    # textbox.on_text_change(lambda v: print(v))
    linebuilder = LineBuilder(line,ax,'red')
    plt.gca().invert_yaxis() # inverse the y axis
    #-------------------------------------------------------------------------------------------------#
    # De dua anh ra full man hinh, khong co cai phan bouding box o ngoai
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #-------------------------------------------------------------------------------------------------#o
    # De hien thi anh full man hinh may tinh 
    manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    manager.window.showMaximized()
    #-------------------------------------------------------------------------------------------------#
    plt.show()
    new_shapes, original_shape = change_shapes(linebuilder.shape)
    return new_shapes, original_shape
#-----------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('side', type=str, help="Indicate that we would like to do the annotation for the Bud - Darken part - Root - or Full bud.")
args = parser.parse_args()

if args.side == 'B':	# Stand for "bud"
    obj = 'bud'
elif args.side == 'L':	# Stand for "darkenpart"
    obj = 'leaves'
elif args.side == 'C':	# Stand for "root"
    obj = 'core'
else:
    obj = 'fullbud'

if __name__ == '__main__':
    filenames = sorted(glob.glob(path + "/*.*")) #read all files in the path mentioned
    init = 129
    filenames = filenames[init:]
    for n, image_file in enumerate(filenames):
        file, ext = os.path.splitext(image_file)  # split filename and extension
        print("file name", file)
        #total_pic = len([f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))])
        total_pic = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        no_images = no_images + 1 
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = image
        new_shapes, shapes = create_shape_on_image(img)
        print(shapes) # OK
        #print(shapes[0][0]) # OK
        #print(shapes[0][0][0]) # OK
        for i in range(len(shapes)):
            #--------------- Angle calculation ------------------------------------------------#
            x1 = shapes[i][0][0]
            y1 = shapes[i][0][1]
            x2 = shapes[i][1][0]
            y2 = shapes[i][1][1]
            x3 = x2
            y3 = y1
            print(x1,y1,x2,y2,x3,y3)
            AB = math.sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1))
            print("AB = ", AB)
            AC = math.sqrt((x3 - x1)*(x3 - x1) + (y3 - y1)*(y3 - y1))
            print("AC = ", AC)
            
            if (y2 == y1): # vector nằm ngang # special case
                if (x2 == x1): # error
                    break
                elif (x2 > x1):
                    angle = 0
                else: # (x2 < x1)
                    angle = 180
            elif (x2 == x1): # vector nằm dọc # special case
                if (y2 == y1): # error
                    break
                elif (y2 > y1):
                    angle = 270
                else: # (y2 < y1)
                    angle = 90
            else: # normal case
                if (x2 > x1) and (y2 < y1): # 1st corner
                    angle = math.acos(AC/AB) * 180 / math.pi
                    print("angle = ", angle)
                elif (x2 < x1) and (y2 < y1): # 2nd corner
                    angle = 180 - math.acos(AC/AB) * 180 / math.pi
                    print("angle = ", angle)
                elif (x2 < x1) and (y2 > y1): # 3rd corner
                    angle = 180 + math.acos(AC/AB) * 180 / math.pi
                    print("angle = ", angle)
                else: # 4th corner
                    angle = 360 - math.acos(AC/AB) * 180 / math.pi
                    print("angle = ", angle)
            angle = int(round(angle, 0))
            print("angle = ", angle)
            #--------------- Angle calculation ------------------------------------------------#
            a = np.reshape(new_shapes[i], (len(new_shapes[i])*len(new_shapes[i][0]), )) 
            polygon = a.tolist()
            img = Image.new('L', (width, height), 0)
            #img = Image.new('L', (1344, 1008), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            mask = numpy.array(img)
            mask_temp = (mask*255).astype('uint8') # sample array
            #print(mask_temp.shape)
            im = Image.fromarray(mask_temp)
            imrgb = Image.merge('RGB', (im,im,im)) # color  image
            filename = os.path.join(annotations_savepath, '{}_{}_{}_{}.png'.format(os.path.basename(file),obj,i, angle))
            imrgb.save(filename)
        print("No. of images had done = :", '{}/{}'.format(no_images,total_pic))
    


