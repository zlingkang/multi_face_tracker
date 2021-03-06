#ifndef __HEAD_POSE_ESTIMATION
#define __HEAD_POSE_ESTIMATION

#include <opencv2/core/core.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <vector>
#include <array>
#include <string>

// Anthropometric for male adult
// Relative position of various facial feature relative to sellion
// Values taken from https://en.wikipedia.org/wiki/Human_head
// X points forward
const static cv::Point3f P3D_SELLION(0., 0.,0.);
const static cv::Point3f P3D_RIGHT_EYE(-20., -65.5,-5.);
const static cv::Point3f P3D_LEFT_EYE(-20., 65.5,-5.);
const static cv::Point3f P3D_RIGHT_EAR(-100., -77.5,-6.);
const static cv::Point3f P3D_LEFT_EAR(-100., 77.5,-6.);
const static cv::Point3f P3D_NOSE(21.0, 0., -48.0);
const static cv::Point3f P3D_STOMMION(10.0, 0., -75.0);
const static cv::Point3f P3D_MENTON(0., 0.,-133.0);



static const int MAX_FEATURES_TO_TRACK=100;

// Interesting facial features with their landmark index
enum FACIAL_FEATURE {
    NOSE=30,
    RIGHT_EYE=36,
    LEFT_EYE=45,
    RIGHT_SIDE=0,
    LEFT_SIDE=16,
    EYEBROW_RIGHT=21,
    EYEBROW_LEFT=22,
    MOUTH_UP=51,
    MOUTH_DOWN=57,
    MOUTH_RIGHT=48,
    MOUTH_LEFT=54,
    SELLION=27,
    MOUTH_CENTER_TOP=62,
    MOUTH_CENTER_BOTTOM=66,
    MENTON=8
};


typedef cv::Matx44d head_pose;

class HeadPoseEstimation {

public:

    HeadPoseEstimation(std::string robot_type);

    //void update(const &dlib::full_object_detection shape);

    cv::Vec3i poseUpdate(const dlib::full_object_detection &shape);

    //std::vector<head_pose> poses() const;

    float focalLength;
    float opticalCenterX;
    float opticalCenterY;

    double yaw, pitch;
    cv::Vec3d translation;

    mutable cv::Mat _debug;

    float trans;
    float transy;

    std::vector<cv::Point2f> axes_;
private:

    //dlib::cv_image<dlib::bgr_pixel> current_image;

    //dlib::frontal_face_detector detector;
    //dlib::shape_predictor pose_model;

    //std::vector<dlib::rectangle> faces;

    //std::vector<dlib::full_object_detection> shapes;
    //dlib::full_object_detection shape;

    /** Return the point corresponding to the dictionary marker.
    */
    cv::Point2f coordsOf(const dlib::full_object_detection &shape, FACIAL_FEATURE feature);
    cv::Point2f coordsOfTrans(const dlib::full_object_detection &shape, FACIAL_FEATURE feature);

    /** Returns true if the lines intersect (and set r to the intersection
     *  coordinates), false otherwise.
     */
    bool intersection(cv::Point2f o1, cv::Point2f p1,
                      cv::Point2f o2, cv::Point2f p2,
                      cv::Point2f &r) const;

};

#endif // __HEAD_POSE_ESTIMATION
