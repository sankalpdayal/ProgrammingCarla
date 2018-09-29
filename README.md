## Introduction
For this project, the goal was to design a complete system that drives a vehicle given a camera feed and location. Once the system is designed and tested on simulator then it will be ported to a hardware capable car.

The system is designed using a ROS and data processing is done using python. Publisher and Subscriber model of ROS is used for communication between different sensors and modules.

### Team Members:
* Sankalp Dayal - sankalpdayal@gmail.com

## Project Goals

The project can be divided roughly into followin sub tasks.
1. Estimating the way points for the car to enable it to drive in a straight path.
2. Define controls depending on the current waypoints.
3. Develop video processing system to detect if traffic light exists and its state.
4. Update the way points such that it stops at red lights.


## Waypoint Updater

The purpose of waypoint updater is to fetch the waypoints ahead of the car from list of base waypoints and using the state of traffic light if exists define next set of waypoints that car should follow. The functionality can be categorized as follows

1. Initialization, the base way points are fetched and stored efficiently in a tree format. This allows fetching of closest waypoint in log n steps instead of n.

2. During run, the waypoints are updated at 10 Hz. Everytime two closest waypoints are fetched and waypoint in front is determined.

3. Using this waypoint next 50 waypoints are determined.

4. All the traffic lights are checked if any is close to the next 50 way points. If found the stop line of the corresponding traffic light is obtained.

5. Depending on state of light, if light is red, then using the stop line, the next set of way points are updated such that the car stops at the stop line.
 
## Controling the Car

Contorlling the car is done by using drive-by-wire (DBW) packages. These are drivers that allow communication with the actuators of the car that control braking, steering, acceleration etc. These packages make it easy to 

estimate the throttle, brake, and steering using the speed and rotation rate of the car. Following are the major concepts implemented

1. To control the thottle of the car, a simple PID controller is implemented that uses velocity error.

2. To make the current velocity smooth, a low pass filter is implemented.

3. Steering is estimated using an inbuilt Yaw controller that uses car specific values like wheel_base , steer_ratio, max_lat_accel, max_steer_angle and 
current angulare velocity and linear velocity.

4. To maintain that decel doesnt cross max limits a check is implemented and brake is estimated using the mass of the car and vehicle radius.

5. There is a flag that maintains the status of DBW state and if it is not enabled, the state is reset.


## Traffic Light Detection

The camera feed from the car is used to determine the state of traffic light. The apporach for Traffic Light Detection can be divided into parts

1. Detection of location of light in the image.

2. Using just the traffic light's image, determine the state of light.

3. Process consecutive images and build confidence on the estimation.

### Detection of location of light in image
This was done by using pre trained existing ssd mobilenet v1 trained on coco dataset obtained from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

Since this was trained to detect multiple kinds of objects traffic light was one of those and its id is 10. If a class is detected the model outputs confidence and bounding boxes. 

Hence whenever image is fed and class 10 is detected with high confidence, the pixels in the bounding box are used for next step of determining the state of light.

### Detection of state of traffic light.
 
For this I used the Machine Learning approach. The details on building the classifier are given [here](https://github.com/sankalpdayal/ProgrammingCarla/blob/master/tl/Traffic_light_detection.ipynb)

Following are the details

1. Pre Processing: The input image was first rescaled to 32x32 pixels. The color format was kept as BGR.

2. Feature Extraction: The image was divided into 3 regions along y axis. This roughly correponds to the regions corresponing to 3 lights.
For each region a histogram was obtained with bin size for 2 for all three channels.  This gives total 18 features. One last feature was added that determines
which region has the maximum intensity among three. If top then the feature value was 0, if center then 1 else 2. 

3. Dataset: I used the [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/node/6132) This is divided into test and train and has images obtained from a video feed of a car. 

Each image has information about light like bounding boxes, state of light and if occluded. Using the boxes, the regions corresponding to only traffic light were exctracted and rescaled to 32x32.

For labels the state of light was used. Red, Yellow and Off lights were given label 0 and green as label 1.

4. Classifier Building: I used  Logisitic Regression as classifier. For training, since dataset was imbalanced I give different weights to the samples. 

5. Accuracy: 

|Dataset | Average | Green Detection | Red Detection |
|----|-----|--------|------|
|Training| 95%|98%|91%|
|Test| 92%|91%|94%|
 

### Processing consecutive images
A simple voting algorithm was used in which if last 3 values if same then the state of light was found to be confident. 
 
## Key Points to Make the Project Work Well

1. Because of hardware limitations, I reduced the refresh rate of Way point updator and DBW both to 10 Hz.
2. There was an inbuilt feature of way point updator C++ code, that updates only when car's error from expected was more than a distance. 
I changed this to update everytime.
3. Storing the base way points in a form of KD tree reduced was a neat concept.
4. Tuning the PID parameters could have been done better and still has some scope of improements.

## Installation Details

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Installation 

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop). 
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space
  
  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.2).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Because of the size constraints of github < 100MB we are not able to 
upload our trained tensorflow model for Object Detection. Please download our trained model [frozen_inference_graph.pb] (https://drive.google.com/uc?id=0B8PSf2JS7ts2VDNSWmI1MVZRSDQ) from Google Drive.

2. Please place downloaded file in the tl_detector ros directory (sdc-capstone/ros/src/tl_detector).

```
3. Launch project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
