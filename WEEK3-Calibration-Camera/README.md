# TITLE: CAMERA CALIBRATION

##  1. [CAMERA CALIBRATION with MATLAB](https://www.mathworks.com/help/vision/ug/using-the-single-camera-calibrator-app.html)


## 2. [CAMERA CALIBRATION with OPENCV (Python)](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

A camera is an integral part of several domains like robotics, space exploration, etc camera is playing a major role. It helps to capture each and every moment and helpful for many analyses. In order to use the camera as a visual sensor, we should know the parameters of the camera. Camera Calibration is nothing but estimating the parameters of a camera, parameters about the camera are required to determine an accurate relationship between a 3D point in the real world and its corresponding 2D projection (pixel) in the image captured by that calibrated camera.

We need to consider both internal parameters like focal length, optical center, and radial distortion coefficients of the lens etc., and external parameters like rotation and translation of the camera with respect to some real world coordinate system.

Required libraries:

OpenCV library in python is a computer vision library, mostly used for image processing, video processing, and analysis, facial recognition and detection, etc.

Numpy is a general-purpose array-processing package. It provides a high-performance multidimensional array object and tools for working with these arrays.

Camera Calibration can be done in a step-by-step approach:

Step 1: First define real world coordinates of 3D points using known size of checkerboard pattern.

Step 2: Different viewpoints of check-board image is captured.

Step 3: findChessboardCorners() is a method in OpenCV and used to find pixel coordinates (u, v) for each 3D point in different images.

Step 4: Then calibrateCamera() method is used to find camera parameters.

It will take our calculated (threedpoints, twodpoints, grayColor.shape[::-1], None, None) as parameters and returns list having elements as Camera matrix, Distortion coefficient, Rotation Vectors, and Translation Vectors. 

Camera Matrix helps to transform 3D objects points to 2D image points and the Distortion Coefficient returns the position of the camera in the world, with the values of Rotation and Translation vectors.

## 3. [CHECKERBOARD](https://markhedleyjones.com/projects/calibration-checkerboard-collection?fbclid=IwAR3IkNAHuvoV4naISlC9z9N20BFNkX5lWI6iNu72BftvaJi9zp7V8nf8_lM)
