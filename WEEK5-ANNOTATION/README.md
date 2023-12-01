# This section will guide you about the dataset annotation and augmentation for deep learning model.
## $\color[RGB]{255,0,0}1. Annotation$

For labeling, we need to install the tools that help us annotate the instance segmentation database such as labelme, COCO-annotator, etc.

### Installation 
- [labelme tool](https://github.com/wkentaro/labelme)

```python
pip install labelme
```
```python
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

### Step-by-step tutorials

Firstly, we should divide your own database into two folders (or three including test folder) including train and validate folder with 80%/20% (or 80/10/10) of scale.

Secondly, we operate the labelme tool by using "labelme" command and open each folder to label all your own databases.

After that, we will convert the json file results of labelme to masks of each object in a image by using the code "convert_json_labelme2mask.py"

```python
python convert_json_labelme2mask.py
```
Then, we should create two folders like the below image (IMG and ANNOTATION). In  each train and validate folder consits of two subfolders which contain images and masks. We rename each it's subfolder to train and validate folder and copy them to IMG and ANNOTATION, respectively. 

![image](https://github.com/ThinhPham24/PRACTICE_ON_COMPUTER_VISION/assets/58129562/97f6d61a-354f-4055-8375-d72dbc06894b)

Finally, we should check the results by using the code "check_only_annotation.py". Let open the code and change the $\color[RGB]{155,127,0}your own path$ . For example is as below.

![image](https://github.com/ThinhPham24/PRACTICE_ON_COMPUTER_VISION/assets/58129562/2ff2b1ce-453b-4efb-9450-f05bafc71a8d)

## $\color[RGB]{255,0,0}2. Augmentation$

We have some augmentation method such as $\color[RGB]{0,255,0}scale$, $\color[RGB]{0,255,0}rotation$, $\color[RGB]{0,255,0}shear$, $\color[RGB]{0,255,0}Gradiennt$, etc. In this tutorial will guide you how to create more databases based on augmentation methods. If you think your own dataset need other method, you can add it into our code.

### Installation
We have  more librarys to augmentate the database such as [ablualbumentations](https://github.com/albumentations-team/albumentations), [aug](https://github.com/tgilewicz/aug), [aug-tool](https://pypi.org/project/aug-tool/0.0.2/), etc.

```python
pip install ablualbumentations
```
```
pip install aug
```
```
pip install aug-tool==0.0.2
```

### Step-by-step tutorials

This tutorial includes some augmentation methods like scale, rotation, linear gradients and random contrast,etc.

#### a. Scale method

We should change the $\color[RGB]{0,255,0}paht$  of $\color[RGB]{0,255,0}folder$ of your own dataset in ```scale.py```.

![image](https://github.com/ThinhPham24/PRACTICE_ON_COMPUTER_VISION/assets/58129562/99d026e8-93e2-4d69-acf2-0711385b8de8)

Run this code

```python
python scale.py
```

#### b. Rotation method

Remenber that you need to change the path of your own dataset.

```python
python Rot_aug.py
```

#### c. Linear Gradient along x, y axis, Random Contrast, Brightness and ColorJitter

Remenber that you need to change the path of your own dataset.

```python
python augmentation.py
```

## $\color[RGB]{255,0,0}3. Converting$ $\color[RGB]{255,0,0} DL$ $\color[RGB]{255,0,0} format$ $\color[RGB]{255,0,0} COCO, YOLO format, VOC,etc.$

This tutorial guide us about converting the Mask of th Object to the COCO Instance Segmentation Format.

Run this code

```python
python mask2coco.py
```

