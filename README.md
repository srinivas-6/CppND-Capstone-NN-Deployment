# CPPND: Capstone Project 

## YOLOv3 Deployment with OpenCV using C++
This is the capstone project for the Udacity C++ Nanodegree Program. YOLO Object detection model using OpenCV is implemented. The program reads an image file given by the user and then performs object detection on it.The output image is displayed with bounding boxes over the detected objects and is also saved into the images folder. This program is  built using the concepts learnt in the nanodegree program. It can also be used for various Deep Neural Network Models.The compiled file is executed through the command line and takes in various options (image / video) as input parameters.

## Dependencies for Running Locally
* cmake >= 3.7
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCv >=4.1
  * For Installation [here](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html)    
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Enter into yolo_model directory. `cd yolo_model`
3. `sudo chmod a+x get_models.sh`
4. `./get_models.sh`
5. Make a build directory in the top level directory: `mkdir build && cd build`
6. Compile: `cmake .. && make`
7. Run it: `./yolo --image=../images/test_image.jpg`
8. If you want to change model parameters Run `./yolo --image=../images/test_image.jpg --conf=0.6 --nms=0.5`
9. You can also give video as input `./yolo --video=path to video file`
