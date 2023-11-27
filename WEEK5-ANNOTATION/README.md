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

First, We have some augmentation method such as $\color[RGB]{0,255,0}scale$, $\color[RGB]{0,255,0}rotation$, $\color[RGB]{0,255,0}shear$, $\color[RGB]{0,255,0}Gradiennt$, etc. In this tutorial will guide you how to create more database based on augmentation methods. If you think your own dataset need other method, you can add it into our code.

### Installation
We have  more 
[ablualbumentations](https://github.com/albumentations-team/albumentations)

```python
pip install ablualbumentations
```
