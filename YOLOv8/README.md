# YOLOv8
Training yolov8 on colab and prediction on PC.

This rep is unofficial. If you are interested in YOLOv8, you can visit official [here](https://github.com/ultralytics/ultralytics).

# Tips for the Best Training Result
📚 This guide explains how to produce the best mAP and training results with YOLOv5 🚀. UPDATED 25 May 2022.

Most of the time good results can be obtained with no changes to the models or training settings, provided your dataset is sufficiently large and well labelled. If at first you don't get good results, there are steps you might be able to take to improve, but we always recommend users first train with all default settings before considering any changes. This helps establish a performance baseline and spot areas for improvement.

If you have questions about your training results we recommend you provide the maximum amount of information possible if you expect a helpful response, including results plots (train losses, val losses, P, R, mAP), PR curve, confusion matrix, training mosaics, test results and dataset statistics images such as labels.png. All of these are located in your project/name directory, typically yolov5/runs/train/exp.

We've put together a full guide for users looking to get the best results on their YOLOv5 trainings below.

Dataset:
- Images per class. ≥ 1500 images per class recommended
- Instances per class. ≥ 10000 instances (labeled objects) per class recommended
- Image variety. Must be representative of deployed environment. For real-world use cases we recommend images from different times of day, different seasons, different weather, different lighting, different angles, different sources (scraped online, collected locally, different cameras) etc.
- Label consistency. All instances of all classes in all images must be labelled. Partial labelling will not work.
- Label accuracy. Labels must closely enclose each object. No space should exist between an object and it's bounding box. No objects should be missing a label.
- Label verification. View train_batch*.jpg on train start to verify your labels appear correct, i.e. see example mosaic.
- Background images. Background images are images with no objects that are added to a dataset to reduce False Positives (FP). We recommend about 0-10% background images to help reduce FPs (COCO has 1000 background images for reference, 1% of the total). No labels are required for background images.

