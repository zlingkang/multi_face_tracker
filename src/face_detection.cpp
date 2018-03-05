#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <actionlib/server/simple_action_server.h>
#include <multi_face_tracker/FaceDetectionAction.h>
#include <unistd.h>

class FaceDetectionAction
{
    protected:
        ros::NodeHandle nh_;
        actionlib::SimpleActionServer<multi_face_tracker::FaceDetectionAction> as_;
        std::string action_name_;

        multi_face_tracker::FaceDetectionFeedback feedback_;

        image_transport::Subscriber image_sub_;
        
        dlib::frontal_face_detector dlib_detector_;

        cv::Mat raw_img_;

        bool pyramid_up_;
    public:
        FaceDetectionAction(std::string name):
            //nh_("~"),
            as_(nh_, name, boost::bind(&FaceDetectionAction::executeCB, this, _1), false),
            dlib_detector_(dlib::get_frontal_face_detector()),
            action_name_(name)
        {
            //nh_ = new ros::NodeHandle("~");
            //as_ = actionlib::SimpleActionServer<multi_face_tracker::FaceDetectionAction>(*nh_, name, boost::bind(&FaceDetectionAction::executeCB, this, _1), false);
            as_.start();
            image_transport::ImageTransport it(nh_);
            std::string cam_type;
            ros::param::get("~cam_type", cam_type);
            if(cam_type == "usb_cam")
            {
                image_sub_ = it.subscribe("/usb_cam_node/image_raw", 10, &FaceDetectionAction::ImageCallback, this);
            }
            else
            {
                image_sub_ = it.subscribe("/bebop/image_raw", 10, &FaceDetectionAction::ImageCallback, this);
            }
            ros::param::get("~pyramid_up", pyramid_up_);
        }
        ~FaceDetectionAction(void)
        {

        }

        void executeCB(const multi_face_tracker::FaceDetectionGoalConstPtr &goal)
        {
            ros::Rate r(30);
            bool success = true;

            multi_face_tracker::FaceDetectionResult result_;
            cv::Mat _cv_img = raw_img_.clone();
            
            feedback_.finished = 0;

            //dlib::cv_image<dlib::bgr_pixel> _dlib_img(_cv_img);
            dlib::array2d<dlib::bgr_pixel> _dlib_img;
            dlib::assign_image(_dlib_img, dlib::cv_image<dlib::bgr_pixel>(_cv_img));
            if(pyramid_up_)
            {
                dlib::pyramid_up(_dlib_img);
            }
            std::vector<dlib::rectangle> _dlib_faces = dlib_detector_(_dlib_img);
            //std::this_thread::sleep_for (std::chrono::seconds(1)); 
            //usleep(300*1000);
            feedback_.finished = 1;
            as_.publishFeedback(feedback_);

            result_.request_id = goal->request_id;
            if(_dlib_faces.size())
            {
                //result_.amount = _dlib_faces.size();
                result_.amount = 0;
                for(int i = 0; i < _dlib_faces.size(); i ++)
                {
                    if(pyramid_up_)
                    {
                        result_.left.push_back(_dlib_faces[i].left()/2);
                        result_.top.push_back(_dlib_faces[i].top()/2);
                        result_.width.push_back(_dlib_faces[i].width()/2);
                        result_.height.push_back(_dlib_faces[i].height()/2);
                    }
                    else
                    {
                        result_.left.push_back(_dlib_faces[i].left());
                        result_.top.push_back(_dlib_faces[i].top());
                        result_.width.push_back(_dlib_faces[i].width());
                        result_.height.push_back(_dlib_faces[i].height());
                       
                    }
                    result_.amount = result_.amount + 1;
                }
            }
            else
            {
                result_.amount = 0;
            }
            if(success)
            {
                as_.setSucceeded(result_);
            }
        }

        void ImageCallback(const sensor_msgs::ImageConstPtr& cam_msg)
        {
            cv_bridge::CvImagePtr cv_ptr;
            try
            {
                cv_ptr = cv_bridge::toCvCopy(cam_msg, sensor_msgs::image_encodings::BGR8);
            }
            catch(cv_bridge::Exception& e)
            {
                ROS_ERROR("Error converting the input image: %s", e.what());
                return;
            }
            raw_img_ = cv_ptr->image;
        }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "face_detection");

    FaceDetectionAction face_detection("face_detection");
    ros::spin();

    return 0;
}
