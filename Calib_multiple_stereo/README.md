# Calibrate_multiple_stereo
Multiple stereo cameras.
## Following these below stages:
1. STAGE 1: Calibrate each stereo camera.
* Step 1. Run the code to collect the images of each stereo
```bash
python3 S1_take_normal_image.py
```
* Step 2. Should change its folder name of each view, respectively.
```python
image_dir = "IMG_VIEWS"
Img_name = "degree0"
```
After that, we will run the code.
```bash
python3 S2_stereo_calibration.py
```
We will get the result like this.
![The results](https://github.com/ThinhPham24/Calib_multiple_stereo/blob/master/IMAGES_GUI/Matrix.jpg)

2. STAGE 2: Find the transformation matrix of each side view (pair camera) to convert to Top view
* Step 1. 
