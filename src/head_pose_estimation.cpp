// modified from https://github.com/chili-epfl/attention-tracker
// input the landmark of a single face and return the translation and pitch, yaw

#include <cmath>
#include <ctime>

#include <math.h>

#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "head_pose_estimation.hpp"

//using namespace dlib;
using namespace std;
using namespace cv;

inline Point2f toCv(const dlib::point& p)
{
    return Point2f(p.x(), p.y());
}


HeadPoseEstimation::HeadPoseEstimation(string robot_type)
{

    focalLength = 300.0;
    opticalCenterX = 320.0;
    if(robot_type == "bebop"){
        opticalCenterX = 428.0;
        opticalCenterY = 239.0;//240.0 //for chatterbox and 184 for Bebop;
    }
    else{
        opticalCenterY = 240.0;
    }
}

cv::Vec3i HeadPoseEstimation::poseUpdate(const dlib::full_object_detection &shape)
{

    cv::Mat projectionMat = cv::Mat::zeros(3,3,CV_32F);
    cv::Matx33f projection = projectionMat;
    projection(0,0) = focalLength;
    projection(1,1) = focalLength;
    projection(0,2) = opticalCenterX;
    projection(1,2) = opticalCenterY;
    projection(2,2) = 1;

    std::vector<Point3f> head_points;

    head_points.push_back(P3D_SELLION);
    head_points.push_back(P3D_RIGHT_EYE);
    head_points.push_back(P3D_LEFT_EYE);
    head_points.push_back(P3D_RIGHT_EAR);
    head_points.push_back(P3D_LEFT_EAR);
    head_points.push_back(P3D_MENTON);
    head_points.push_back(P3D_NOSE);
    head_points.push_back(P3D_STOMMION);

    std::vector<Point2f> detected_points;

    detected_points.push_back(coordsOf(shape, SELLION));
    detected_points.push_back(coordsOf(shape, RIGHT_EYE));
    detected_points.push_back(coordsOf(shape, LEFT_EYE));
    detected_points.push_back(coordsOf(shape, RIGHT_SIDE));
    detected_points.push_back(coordsOf(shape, LEFT_SIDE));
    detected_points.push_back(coordsOf(shape, MENTON));
    detected_points.push_back(coordsOf(shape, NOSE));

    auto stomion = (coordsOf(shape, MOUTH_CENTER_TOP) + coordsOf(shape, MOUTH_CENTER_BOTTOM)) * 0.5;
    detected_points.push_back(stomion);

    Mat rvec0;
    Mat tvecTrans;

    Mat tvec = (Mat_<double>(3,1) << 0., 0., 1000.);
    Mat rvec = (Mat_<double>(3,1) << 1.2, 1.2, -1.2);

    // face center x coordinate
    trans = stomion.x - 320.0;
    transy = stomion.y - 184.0; //240.0;
    std::vector<Point2f> trans_points;
    trans_points.push_back(coordsOfTrans(shape, SELLION));
    trans_points.push_back(coordsOfTrans(shape, RIGHT_EYE));
    trans_points.push_back(coordsOfTrans(shape, LEFT_EYE));
    trans_points.push_back(coordsOfTrans(shape, RIGHT_SIDE));
    trans_points.push_back(coordsOfTrans(shape, LEFT_SIDE));
    trans_points.push_back(coordsOfTrans(shape, MENTON));
    trans_points.push_back(coordsOfTrans(shape, NOSE));
    auto stomionTrans = (coordsOfTrans(shape, MOUTH_CENTER_TOP) + coordsOfTrans(shape, MOUTH_CENTER_BOTTOM)) * 0.5;
    trans_points.push_back(stomionTrans);


    // Find the 3D pose of our head, we get the rotation after we move the face interesting points to the center of the image
    solvePnP(head_points, detected_points,
            projection, noArray(),
            rvec0, tvec, false,
            cv::SOLVEPNP_ITERATIVE);

    solvePnP(head_points, trans_points, projection, noArray(), rvec, tvecTrans, false, cv::SOLVEPNP_ITERATIVE);

    Matx33d rotation;

    
    cout << "start rodrigues" << endl;
    Rodrigues(rvec, rotation);
    cout << "end rodrigues" << endl;

/*
    Vec3d euler;
    Matx33d mtxR, mtxQ;
    euler = RQDecomp3x3(rotation, mtxR, mtxQ);
    cout << "euler" << euler << endl;
*/
    //Computing Euler angles from a rotation matrix
    //Gregory G. Slabaugh
    if(rotation(2,0) == 1){
        rotation(2,0) -= 0.001;
    }
    else if(rotation(2,0) == -1){
        rotation(2,0) += 0.001;
    }
    double theta1 = (- asin(rotation(2,0)));
    double theta2 = 3.14 - theta1;
    double phi1 = atan2(rotation(2,1)/cos(theta1), rotation(2,2)/cos(theta1));
    double phi2 = atan2(rotation(2,1)/cos(theta2), rotation(2,2)/cos(theta2));
    double thi1 = atan2(rotation(1,0)/cos(theta1), rotation(0,0)/cos(theta1));
    double thi2 = atan2(rotation(1,0)/cos(theta2), rotation(0,0)/cos(theta2));
    
    cout << theta1 << " " << phi1 << " " << thi1 << endl;
    cout << theta2 << " " << phi2 << " " << thi2 << endl;
    
    head_pose pose = {
        rotation(0,0),    rotation(0,1),    rotation(0,2),    tvec.at<double>(0)/1000,
        rotation(1,0),    rotation(1,1),    rotation(1,2),    tvec.at<double>(1)/1000,
        rotation(2,0),    rotation(2,1),    rotation(2,2),    tvec.at<double>(2)/1000,
                    0,                0,                0,                     1
    };
    
    // get the translation in centimeter
    translation[0] = tvec.at<double>(0)/10.0;
    translation[1] = tvec.at<double>(1)/10.0;
    translation[2] = tvec.at<double>(2)/10.0;

    //std::cout << "translation: " << translation << std::endl;

    std::vector<Point2f> reprojected_points;

    projectPoints(head_points, rvec, tvec, projection, noArray(), reprojected_points);

    std::vector<Point3f> axes;
    axes.push_back(Point3f(0,0,0));
    axes.push_back(Point3f(50,0,0));
    axes.push_back(Point3f(0,50,0));
    axes.push_back(Point3f(0,0,50));
    std::vector<Point2f> projected_axes;

    projectPoints(axes, rvec0, tvec, projection, noArray(), projected_axes);

    /*
    line(_debug, projected_axes[0], projected_axes[3], Scalar(255,0,0),2,CV_AA); // Blue
    line(_debug, projected_axes[0], projected_axes[2], Scalar(0,255,0),2,CV_AA); // Green
    line(_debug, projected_axes[0], projected_axes[1], Scalar(0,0,255),2,CV_AA); // Red
    */

    axes_.clear();
    for(int i = 0; i < 4; i ++){
        axes_.push_back(projected_axes[i]);
    }

    // print pitch and yaw
    if (projected_axes[1].x > projected_axes[0].x){
        yaw =  theta2 - 3.14/2.0;
        pitch = thi2;
    }
    else{
        yaw =  theta1 - 3.14/2.0;
        pitch = -thi1;
    }
    //cout << "yaw:" << yaw*180.0/3.14 << " pitch:" << pitch*180.0/3.14 << endl; 

    putText(_debug, "(" + to_string(int(pose(0,3) * 100)) + "cm, " + to_string(int(pose(1,3) * 100)) + "cm, " + to_string(int(pose(2,3) * 100)) + "cm)", coordsOf(shape, SELLION), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255),2);

    cv::Vec3i pose_final;//( (int)(yaw*180/3.14), (int)translation[1], (int)translation[2] );
    pose_final[0] = (int)(yaw*180/3.14);
    pose_final[1] = -(int)translation[2];
    pose_final[2] = (int)translation[0];

    return pose_final;
}

Point2f HeadPoseEstimation::coordsOf(const dlib::full_object_detection &shape, FACIAL_FEATURE feature)
{
    return toCv(shape.part(feature));
}
Point2f HeadPoseEstimation::coordsOfTrans(const dlib::full_object_detection &shape, FACIAL_FEATURE feature)
{
    Point2f p =  toCv(shape.part(feature));
    p.x = p.x - trans;
    p.y = p.y - transy;
    return p;
}


// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
// taken from: http://stackoverflow.com/a/7448287/828379
bool HeadPoseEstimation::intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
                                      Point2f &r) const
{
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

