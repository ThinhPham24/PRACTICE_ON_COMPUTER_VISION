# Object detection based on Color by Opencv

### Algorithm

- Using HSV inrange to filter background out;
- Find contours and min rectangel box of the contour(Example case biggest one); 
- Then put all pixel in the contour into black image. So we get mask image;
- Rotate and crop to be final image;

### How to use the code

- Setting HSV lower bound and Upper bound in (HSV_inrange.py)
- Replace those value in function img_process
- set input path in function main
