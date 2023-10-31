# TITLE: CAMERA CALIBRATION

##  1. [CAMERA CALIBRATION with MATLAB](https://www.mathworks.com/help/vision/ug/using-the-single-camera-calibrator-app.html)

You can use the Camera Calibrator app to estimate camera intrinsics, extrinsics, and lens distortion parameters. You can use these camera parameters for various computer vision applications. These applications, such as removing the effects of lens distortion from an image, measuring planar objects, or reconstructing 3-D scenes from multiple cameras.

The suite of calibration functions used by the Camera Calibrator app provide the workflow for camera calibration. You can use these functions directly in the MATLAB® workspace. For a list of functions, see Camera Calibration.

Workflow diagram ; prepare images > add images, > calibrate > evaluate > improve > export.

Follow this workflow to calibrate your camera using the app.

Prepare the images, camera, and calibration pattern.

Add the images and select standard or fisheye camera model.

Calibrate the camera.

Evaluate the calibration accuracy.

Adjust the parameters to improve the accuracy (if necessary).

Export the parameters object.

If the default values work well, then you do not need to make any adjustments before exporting the parameters.

Choose a Calibration Pattern
The Camera Calibrator app supports checkerboard, circle grid, and custom detector patterns. For details on each of these patterns and PDF files containing printable patterns, see Calibration Patterns.

Capture Calibration Images
For best calibration results, use between 10 and 20 images of the calibration pattern. The calibrator requires at least three images. Use uncompressed images or lossless compression formats such as PNG. The calibration pattern and the camera setup must satisfy a set of requirements to work with the calibrator. For more details on camera setup and capturing images, see Prepare Camera and Capture Images.

Using the Camera Calibrator App
Open the App
MATLAB Toolstrip: On the Apps tab, in the Image Processing and Computer Vision section, click the Camera Calibrator icon.

MATLAB command prompt: Enter cameraCalibrator

Add Images and Select Camera Model
To begin calibration, you must add images. You can add saved images from a folder or add images directly from a camera. The calibrator analyzes the images to ensure they meet the calibrator requirements. The calibrator then detects the points of the selected pattern. For details on camera setup and capturing images, see Prepare Camera and Capture Images.

 Add Images from File

 Acquire Live Images

After you add images, the Image and Pattern Properties dialog box to your session, appears. Before the calibrator can analyze the calibration patterns, you must select the calibration pattern to detect and set image properties for the pattern structure. For more details on this dialog, see Select Calibration Pattern and Set Properties.

 Analyze Images

 View Images and Detected Points

Calibrate
Once you are satisfied with the accepted images, on the Calibration tab, select Calibrate. The default calibration settings use a minimum set of camera parameters. Start by running the calibration with the default settings. After evaluating the results, you can try to improve calibration accuracy by adjusting the settings or adding or removing images, and then calibrating again. If you switch between the standard and fisheye camera models, you must recalibrate.

 Select Camera Model

 Standard Model Options

 Fisheye Model Options

 Calibration Algorithm

Evaluate Calibration Results
You can evaluate calibration accuracy by examining the reprojection errors, examining the camera extrinsics, or viewing the undistorted image. For best calibration results, use all three methods of evaluation.

Camera calibration results, displaying undistorted image, reprojection errors chart, and camera extrinsics diagram

 Examine Reprojection Errors

 Examine Extrinsic Parameter Visualization

Click to Collapse View Undistorted Image

To view the effects of removing lens distortion, on the Calibration tab, in the View section, select Show Undistorted in the View section of the Calibration tab. If the calibration is accurate, the distorted lines in the image preview become straight.

Original image and undistorted image.

Note

Checking the undistorted images is important even if the reprojection errors are low. For example, if the pattern covers only a small percentage of the image, the distortion estimation can be incorrect, even though the calibration resulted in few reprojection errors. This image shows an example of this type of incorrect estimation for a single camera calibration.

Original image and incorrectly undistorted image.

For the fisheye camera model, while viewing the undistorted images, you can examine the fisheye images more closely by, on the Calibration tab, in the View section, specifying the Fisheye Scale. Enter a value in the Fisheye Scale box, or use the arrows to adjust the scale up or down.

Fisheye scale setting.

Improve Calibration
To improve the calibration, you can remove high-error images, add more images, or modify the calibrator settings.

Consider adding more images if:

You have fewer than 10 images.

The calibration patterns do not cover enough of the image frame.

The calibration patterns do not have enough variation in orientation with respect to the camera.

Consider removing images if the images:

Have a high mean reprojection error.

Are blurry.

Contain a calibration pattern at an angle greater than 45 degrees relative to the camera plane.

Calibration pattern at angle greater than 45 degrees to the camera plane.

Incorrectly detected calibration pattern points.

 Standard Model: Change the Number of Radial Distortion Coefficients

 Standard Model: Compute Skew

 Standard Model: Compute Tangential Distortion

 Fisheye Model: Estimate Alignment

Export Camera Parameters
When you are satisfied with your calibration accuracy, select Export Camera Parameters for a standard camera model or Export Camera Parameters for a fisheye camera model. You can either export the camera parameters to an object in the MATLAB workspace or generate the camera parameters as a MATLAB script.

 Export Camera Parameters

 Generate MATLAB Script

References
[1] Zhang, Z. “A Flexible New Technique for Camera Calibration.” IEEE Transactions on Pattern Analysis and Machine Intelligence. 22, no. 11 (November 2000): 1330–34. https://doi.org/10.1109/34.888718.

[2] Heikkila, J., and O. Silven. “A Four-step Camera Calibration Procedure with Implicit Image Correction.” In Proceedings of IEEE Computer Society Conference on Computer Vision and Pattern Recognition. 1106–12. San Juan, Puerto Rico: IEEE Comput. Soc, 1997. https://doi.org/10.1109/CVPR.1997.609468.

[3] Scaramuzza, Davide, Agostino Martinelli, and Roland Siegwart. "A Toolbox for Easily Calibrating Omnidirectional Cameras." In Proceedings of IEEE International Workshop on Intelligent Robots and Systems 2006 (IROS 2006), 5695–701. Beijing, China: IEEE, 2006. https://doi.org/10.1109/IROS.2006.282372

[4] Urban, Steffen, Jens Leitloff, and Stefan Hinz. “Improved Wide-Angle, Fisheye and Omnidirectional Camera Calibration.” ISPRS Journal of Photogrammetry and Remote Sensing 108 (October 2015): 72–79. https://doi.org/10.1016/j.isprsjprs.2015.06.005.

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

