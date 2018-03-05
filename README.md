# multi_face_tracker  

This is a ROS package that detects and tracks multiple faces based on Kalman Filter.  

## Background  
Object detection is always slow, especially on robots based on embedded platforms (for example Raspberry Pi). So I put the face detection task (which is slow) in a ROS action node, which runs at a low frequency. And I keeps tracking every detected faces using Kalman Filter. So that I can do something interesting at higher frequency given all the face bounding boxes, for example to estimate each face's pose.   

## Dependency  
* ROS with opencv package
* Dlib

## Usage  
```catkin build```  
```roslaunch multi_face_tracker usbcam_demo.launch```    

## Credits  
* Dlib created by Davis King, Dlib is used here for face detection and face pose estimation  
* Chili-epfl's code for face pose estimation: https://github.com/chili-epfl/attension-tracker

## License  
You can use it for free for any purposes at your own risk.
