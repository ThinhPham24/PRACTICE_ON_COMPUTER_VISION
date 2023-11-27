# This section will guide you about the dataset annotation and augmentation for deep learning model.
## 1. Annotation

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

Secondly, we operate the labelme tool by using "labelme" commond and open each folder to label all your own databases.

After that, we will convert the json file results of labelme to masks of each object in a image by using the code "convert_json_labelme2mask.py"

```python
python convert_json_labelme2mask.py
```
