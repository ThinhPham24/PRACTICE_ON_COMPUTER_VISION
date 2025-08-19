# TITLE: CAMERA CALIBRATION

##  1. [CAMERA CALIBRATION with MATLAB](https://www.mathworks.com/help/vision/ug/using-the-single-camera-calibrator-app.html)


## 2. [CAMERA CALIBRATION with OPENCV (Python)](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

A camera is an integral part of several domains like robotics, space exploration, etc camera is playing a major role. It helps to capture each and every moment and helpful for many analyses. In order to use the camera as a visual sensor, we should know the parameters of the camera. Camera Calibration is nothing but estimating the parameters of a camera, parameters about the camera are required to determine an accurate relationship between a 3D point in the real world and its corresponding 2D projection (pixel) in the image captured by that calibrated camera.

We need to consider both internal parameters like focal length, optical center, and radial distortion coefficients of the lens etc., and external parameters like rotation and translation of the camera with respect to some real world coordinate system.

Required libraries:

OpenCV library in python is a computer vision library, mostly used for image processing, video processing, and analysis, facial recognition and detection, etc.

Numpy is a general-purpose array-processing package. It provides a high-performance multidimensional array object and tools for working with these arrays.

Camera Calibration can be done in a step-by-step approach:

Implementation:

''' Input: Asymmetrical circular grid image is used in the code below as input. No measurement of the circular unit size is necessary in       circular grid. Fewer pictures are needed than other objects supported by opencv.

Input images link : https://github.com/PawanKrGunjan/Image-Processing/tree/main/calibrateCamera/Images '''

Step 1: The opencv and numpy libraries are imported and the termination criteria to stop the iteration (to be used further in code) is declared.

Step 2: A vector for real world coordinates of the circular grid is created. As measurement of actual circular unit is not needed, so vector is appended with random grid values. Also obj_points and img_points vectors are created to store 3D and 2D points of input image.

Step 3: The distorted image is then loaded and a grayscale version of image is created.

Step 4: Either the cv::findChessboardCorners or the cv::findCirclesGrid function can be used, depending on the type of input (chessboard or circular grid) to get the position of pattern by passing the current image and the board size. Boolean value is returned to indicate if the pattern was included in the input. If its true, 3D object points are updated.

Step 5: When multiple images are used as input, similar equations might get created during calibration which might not give optimal corner detection. Hence cv.cornerSubPix() function analyses images and corners to give better results. Since the algorithm is iterative, we must define the termination criteria (such as the number of iterations and/or accuracy). In circular grids, this function is not always necessary.

Step 6: The 2D image points are also updated from the optimal corner values. Then, using a call to drawChessboardCorners() that inputs our image, corner measurements, and points that were detected are drawn and saved as output image.

Step 7: The calibrateCamera() function is called with the required parameters and output is displayed.

Step 8: Finally, the error, the camera matrix, distortion coefficients, rotation matrix and translation matrix is printed.

It will take our calculated (threedpoints, twodpoints, grayColor.shape[::-1], None, None) as parameters and returns list having elements as Camera matrix, Distortion coefficient, Rotation Vectors, and Translation Vectors. 

Camera Matrix helps to transform 3D objects points to 2D image points and the Distortion Coefficient returns the position of the camera in the world, with the values of Rotation and Translation vectors.

## 3. [CHECKERBOARD](https://markhedleyjones.com/projects/calibration-checkerboard-collection?fbclid=IwAR3IkNAHuvoV4naISlC9z9N20BFNkX5lWI6iNu72BftvaJi9zp7V8nf8_lM)
