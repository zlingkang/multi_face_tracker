#include <string>
#include <vector>
#include <cmath>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/Twist.h"
#include "image_transport/image_transport.h"
#include "actionlib/client/simple_action_client.h"
#include "multi_face_tracker/FaceDetectionAction.h"

#include "head_pose_estimation.hpp"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

/***********************************FaceTracker Class****************************************/
class FaceTracker
{
    public:
        FaceTracker(int32_t _tracker_id, int32_t _min_accept_frames, int32_t _min_reject_frames, int32_t _image_width, int32_t _image_height);
        ~FaceTracker();
        void KalmanUpdate(const cv::Rect _det_face);
        void KalmanPredict(cv::Rect &_pred_rect);
        cv::Rect FaceROI();
        void DetectFaceInROI(const cv::Mat* _img_ptr);
        double left() const;
        double top() const;
        double width() const;
        double height() const;
        void miss();
        bool accepted() const;
        bool rejected() const;
        int32_t id() const;
        int32_t found_frames_;
        std::vector<cv::Point2f> axes_;
        dlib::full_object_detection landmark_;
        bool has_landmark_;

    
    private:
        int32_t ID_;

        int32_t IMAGE_WIDTH_;
        int32_t IMAGE_HEIGHT_;

        cv::KalmanFilter kf_;
        cv::Mat kf_state_; //x,y,xdot,ydot,w,h
        cv::Mat kf_measure_; 
        float p_cov_scalar_;
        float m_cov_scalar_;

        int32_t MIN_ACCEPT_FRAMES_;
        int32_t MIN_REJECT_FRAMES_;
        bool accepted_;
        bool rejected_;
        int32_t missing_frames_;

        // whether this is the first time to get a detection from the detector
        bool first_time_;

        double ticks_;

        cv::Rect pred_rect_;

        cv::Rect roi_rect_;

        dlib::frontal_face_detector dlib_detector_;

};

int32_t FaceTracker::id() const
{
    return ID_;
}

double FaceTracker::left() const
{
    return pred_rect_.x;
}

double FaceTracker::top() const
{
    return pred_rect_.y;
}

double FaceTracker::width() const
{
    return pred_rect_.width;
}

double FaceTracker::height() const
{
    return pred_rect_.height;
}

void FaceTracker::miss()
{
    missing_frames_ ++;
    if(missing_frames_ > MIN_REJECT_FRAMES_)
    {
        rejected_ = true;
    }
}

bool FaceTracker::accepted() const
{
    return accepted_;
}

bool FaceTracker::rejected() const
{
    return rejected_;
}

// Kalman Filter states number is 6, measurement number is 4
FaceTracker::FaceTracker(int32_t _tracker_id, int32_t _min_accept_frames, int32_t _min_reject_frames, int32_t _image_width, int32_t _image_height)
    :ID_(_tracker_id),
    kf_(6, 4, 0),
    kf_state_(6, 1, CV_32F),
    kf_measure_(4, 1, CV_32F),
    MIN_ACCEPT_FRAMES_(_min_accept_frames),
    MIN_REJECT_FRAMES_(_min_reject_frames),
    found_frames_(0),
    missing_frames_(0),
    IMAGE_WIDTH_(_image_width),
    IMAGE_HEIGHT_(_image_height),
    first_time_(true),
    accepted_(false),
    rejected_(false),
    pred_rect_ (0, 0, 1, 1),
    has_landmark_(false),
    dlib_detector_(dlib::get_frontal_face_detector())
{
    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(kf_.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf_.measurementMatrix = cv::Mat::zeros(4, 6, CV_32F);
    kf_.measurementMatrix.at<float>(0) = 1.0f;
    kf_.measurementMatrix.at<float>(7) = 1.0f;
    kf_.measurementMatrix.at<float>(16) = 1.0f;
    kf_.measurementMatrix.at<float>(23) = 1.0f;
   
    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf_.processNoiseCov.at<float>(0) = 0.05; //1e-2;
    kf_.processNoiseCov.at<float>(7) = 0.05; //1e-2;
    kf_.processNoiseCov.at<float>(14) = 0.05; //5.0f;
    kf_.processNoiseCov.at<float>(21) = 0.05; //5.0f;
    kf_.processNoiseCov.at<float>(28) = 0.05; //1e-2;
    kf_.processNoiseCov.at<float>(35) = 0.05; //1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar(1e-1));
    
    ticks_ = cv::getTickCount();

}

FaceTracker::~FaceTracker()
{

}

void FaceTracker::KalmanUpdate(cv::Rect _det_face)
{
    if(!accepted_)
    {
        found_frames_ ++;
        if(found_frames_ >= MIN_ACCEPT_FRAMES_)
        {
            accepted_ = true;
        }
    }
    missing_frames_ = 0;

    kf_measure_.at<float>(0) = _det_face.x + _det_face.width/2;
    kf_measure_.at<float>(1) = _det_face.y + _det_face.height/2;
    kf_measure_.at<float>(2) = _det_face.width;
    kf_measure_.at<float>(3) = _det_face.height;

    if(first_time_)
    {
        first_time_ = false;
        kf_.errorCovPre.at<float>(0) = 1; // px
        kf_.errorCovPre.at<float>(7) = 1; // px
        kf_.errorCovPre.at<float>(14) = 1;
        kf_.errorCovPre.at<float>(21) = 1;
        kf_.errorCovPre.at<float>(28) = 1; // px
        kf_.errorCovPre.at<float>(35) = 1; // px

        kf_state_.at<float>(0) = kf_measure_.at<float>(0);
        kf_state_.at<float>(1) = kf_measure_.at<float>(1);
        kf_state_.at<float>(2) = 0;
        kf_state_.at<float>(3) = 0;
        kf_state_.at<float>(4) = kf_measure_.at<float>(2);
        kf_state_.at<float>(5) = kf_measure_.at<float>(3);
        // <<<< Initialization

        kf_.statePost = kf_state_;
    }
    else
    {
        kf_.correct(kf_measure_);
    }
}

void FaceTracker::KalmanPredict(cv::Rect &_pred_rect)
{
    double prec_tick = ticks_;
    ticks_ = (double) cv::getTickCount();
    double dT = (ticks_ - prec_tick) / cv::getTickFrequency();
    kf_.transitionMatrix.at<float>(2) = dT;
    kf_.transitionMatrix.at<float>(9) = dT;
    kf_state_ = kf_.predict();
    
    _pred_rect.width = kf_state_.at<float>(4);
    _pred_rect.height = kf_state_.at<float>(5);
    _pred_rect.x = kf_state_.at<float>(0) - _pred_rect.width/2;
    _pred_rect.y = kf_state_.at<float>(1) - _pred_rect.height / 2;
    if(_pred_rect.x < 0)
    {
        _pred_rect.x = 0;
    }
    if(_pred_rect.y < 0)
    {
        _pred_rect.y = 0;
    }
    if((_pred_rect.y + _pred_rect.height >= IMAGE_HEIGHT_))
    {
        if(_pred_rect.y >= IMAGE_HEIGHT_)
        {
            _pred_rect.y = IMAGE_HEIGHT_ - 2;
            _pred_rect.height = 1;
        }
        _pred_rect.height = IMAGE_HEIGHT_ - _pred_rect.y - 1;
    }
    if((_pred_rect.x + _pred_rect.width >= IMAGE_WIDTH_))
    {
        if(_pred_rect.x >= IMAGE_WIDTH_)
        {
            _pred_rect.x = IMAGE_WIDTH_ - 2;
            _pred_rect.width = 1;
        }
        _pred_rect.width = IMAGE_WIDTH_ - _pred_rect.x - 1;
    }
    pred_rect_ = _pred_rect;
    
}

cv::Rect FaceTracker::FaceROI()
{
    roi_rect_.x = pred_rect_.x - pred_rect_.width/2.0;
    roi_rect_.y = pred_rect_.y - pred_rect_.height/2.0;
    roi_rect_.width = 2 * pred_rect_.width;
    roi_rect_.height = 2 * pred_rect_.width;
    if(roi_rect_.x < 0)
    {
        roi_rect_.x = 0;
    }
    if(roi_rect_.y < 0)
    {
        roi_rect_.y = 0;
    }
    if((roi_rect_.y + roi_rect_.height >= IMAGE_HEIGHT_))
    {
        if(roi_rect_.y >= IMAGE_HEIGHT_)
        {
            roi_rect_.y = IMAGE_HEIGHT_ - 2;
            roi_rect_.height = 1;
        }
        roi_rect_.height = IMAGE_HEIGHT_ - roi_rect_.y - 1;
    }
    if((roi_rect_.x + roi_rect_.width >= IMAGE_WIDTH_))
    {
        if(roi_rect_.x >= IMAGE_WIDTH_)
        {
            roi_rect_.x = IMAGE_WIDTH_ - 2;
            roi_rect_.width = 1;
        }
        roi_rect_.width = IMAGE_WIDTH_ - roi_rect_.x - 1;
    }
    return roi_rect_;
}

void FaceTracker::DetectFaceInROI(const cv::Mat *_img_ptr)
{
    cv::Mat cv_img = (*_img_ptr).clone();
    dlib::cv_image<dlib::bgr_pixel> dlib_img(cv_img(roi_rect_));
    std::vector<dlib::rectangle> dlib_faces = dlib_detector_(dlib_img);
    ROS_INFO("Got %d faces in ROI.", dlib_faces.size());
    if(dlib_faces.size())
    {
        cv::Rect det_face;
        det_face.x = dlib_faces[0].left() + roi_rect_.x;
        det_face.y = dlib_faces[0].top() + roi_rect_.y;
        det_face.width = dlib_faces[0].width();
        det_face.height = dlib_faces[0].height();
        this->KalmanUpdate(det_face);
    }
}
/***********************************FaceCtrl Class****************************************/
class FaceCtrl
{
    public:
        FaceCtrl();
        ~FaceCtrl();
        void ImgCallback(const sensor_msgs::ImageConstPtr& cam_msg);
        // get face detection results from face detection action server
        void FaceDetectionCallback(const actionlib::SimpleClientGoalState&, const multi_face_tracker::FaceDetectionResultConstPtr&);
        // update all the face trackers information after getting face detection result from face detection action server
        void FaceTrackersUpdate(std::vector<cv::Rect>_detected_faces);
        double FaceDistance(const FaceTracker _face_tracker, const cv::Rect _detecte_face);
        // Use existing information for prediction between two continuous detections
        void FaceTrackersPredict();

        void FaceDetectInTrackersROI();

        void FacePoseEstimation();

        void FaceTrackersRemove();
    private:
        ros::NodeHandle *node_;
        image_transport::Subscriber image_sub_;
        image_transport::Publisher image_pub_;
        cv::Mat raw_img_;

        //dlib::image_window dlib_win_;
        //dlib::frontal_face_detector face_detector_;
        dlib::shape_predictor face_pose_model_;
        HeadPoseEstimation* estimator_;
        
        actionlib::SimpleActionClient<multi_face_tracker::FaceDetectionAction> face_det_cl_;  
        multi_face_tracker::FaceDetectionGoal face_det_goal_;
        int face_det_id_;

        //whether the first image has arrived
        bool first_image_;

        std::vector<FaceTracker> face_trackers_;
        int32_t face_tracker_id_;

        int image_width_;
        int image_height_;

        int min_accept_;
        int min_reject_;

        std::string cam_type_;
};

FaceCtrl::FaceCtrl():
    face_det_cl_("face_detection", true),
    face_det_id_(0),
    face_tracker_id_(0),
    first_image_(false)
{
    node_ = new ros::NodeHandle("~");

    //image_trans_ = new image_transport::ImageTransport(*node_); 
    //image_pub_ = image_trans_->advertise("inpired_face/face_detection_image", 1);

    estimator_ = new HeadPoseEstimation(cam_type_);

    std::string face_landmark_path;
    node_->getParam("face_landmark_path", face_landmark_path);
    try{
        dlib::deserialize(face_landmark_path) >> face_pose_model_;
    }
    catch(dlib::serialization_error& e){
        std::cout << "need landmarking model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << std::endl;
        std::cout << std::endl << e.what() << std::endl;
    }

    node_->getParam("image_width", image_width_);
    node_->getParam("image_height", image_height_);
    node_->getParam("cam_type", cam_type_);
    node_->getParam("min_accept", min_accept_);
    node_->getParam("min_reject", min_reject_);


    image_transport::ImageTransport it(*node_);
    
    std::string cam_path;
    if(cam_type_ == "usb_cam")
    {
        cam_path = "/usb_cam_node/image_raw";
    }
    else
    {
        cam_path = "/bebop/image_raw";
    }
    image_sub_ = it.subscribe(cam_path, 10, &FaceCtrl::ImgCallback, this);
    image_pub_ = it.advertise("face_detection_image/image_raw", 1);
    
    ROS_INFO("Waiting for face detection action server to start.");
    face_det_cl_.waitForServer();
    ROS_INFO("Face action server started.");
 
}

FaceCtrl::~FaceCtrl(){
     
}

void FaceCtrl::ImgCallback(const sensor_msgs::ImageConstPtr& cam_msg)
{
    //ROS_INFO("Got image!");
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

    if(! first_image_)
    {
        ROS_INFO("start detecting faces in the first image");
        first_image_ = true;
        face_det_goal_.request_id = face_det_id_;
        face_det_id_ ++;
        typedef actionlib::SimpleActionClient<multi_face_tracker::FaceDetectionAction> Client;
        face_det_cl_.sendGoal(face_det_goal_, boost::bind(&FaceCtrl::FaceDetectionCallback, this, _1, _2), Client::SimpleActiveCallback(), Client::SimpleFeedbackCallback());
    }
}

void FaceCtrl::FaceDetectionCallback(const actionlib::SimpleClientGoalState& state, const multi_face_tracker::FaceDetectionResultConstPtr& result)
{
    ROS_INFO("Detected %d faces in frame %d", result->amount, result->request_id);
    face_det_id_ = face_det_id_ % 1000;
    face_det_goal_.request_id = face_det_id_;
    face_det_id_ ++;
    std::vector<cv::Rect> detected_faces;
    for(int i = 0; i < result->amount; i ++)
    {
        cv::Rect face;
        face.x = result->left[i];
        face.y = result->top[i];
        face.width = result->width[i];
        face.height = result->height[i];
        detected_faces.push_back(face);
    }
    FaceTrackersUpdate(detected_faces);
    typedef actionlib::SimpleActionClient<multi_face_tracker::FaceDetectionAction> Client;
    face_det_cl_.sendGoal(face_det_goal_, boost::bind(&FaceCtrl::FaceDetectionCallback, this, _1, _2), Client::SimpleActiveCallback(), Client::SimpleFeedbackCallback());
}


void FaceCtrl::FaceTrackersUpdate(std::vector<cv::Rect> _detected_faces)
{
    // if a detected face is matched with an exisiting tracker
    std::vector<int> face_matched(_detected_faces.size(), 0);

    // match all the existing trackers with the new detected faces
    for(int i = 0; i < face_trackers_.size(); i ++)
    {
        bool tracker_matched = false;
        int32_t matched_face_order = -1;
        double max_distance = 2 * face_trackers_[i].width();
        for(int j = 0; j < _detected_faces.size(); j ++)
        {
            if(face_matched[j])
            {
                continue;
            }
            double face_distance = FaceDistance(face_trackers_[i], _detected_faces[j]);  
            if( face_distance < max_distance)
            {
                matched_face_order = j;
                tracker_matched = true;
                max_distance = face_distance; 
            }
        }
        // Update the kalman filters of the matched trackers
        if(tracker_matched)
        {
            face_matched[matched_face_order] = 1;
            face_trackers_[i].KalmanUpdate(_detected_faces[matched_face_order]);
            //ROS_INFO("Face tracker %d matched", face_trackers_[i].id());
        }
        else
        {
            face_trackers_[i].miss();
            //ROS_INFO("Face tracker %d missed", face_trackers_[i].id());
        }
    }

/*
    for(int i = 0; i < face_trackers_.size(); i ++)
    {
        if(face_trackers_[i].rejected())
        {
            face_trackers_.erase(face_trackers_.begin()+i);
        }
    }
*/
    // Initialize new trackers
    for(int i = 0; i < _detected_faces.size(); i ++)
    {
        if(!face_matched[i])
        {
            //ROS_INFO("Initialize new tracker");
            FaceTracker face_tracker(face_tracker_id_, min_accept_, min_reject_, image_width_, image_height_);
            face_tracker.KalmanUpdate(_detected_faces[i]);
            cv::Rect face_predict;
            face_tracker.KalmanPredict(face_predict);
            face_trackers_.push_back(face_tracker);
            face_tracker_id_ ++;
            face_tracker_id_ = face_tracker_id_ % 1000;
        }
    }
    //ROS_INFO("face trackers number:%d", face_trackers_.size());
}

double FaceCtrl::FaceDistance(const FaceTracker _face_tracker, const cv::Rect _detected_face)
{
    double x1 = _face_tracker.left() + _face_tracker.width()/2.0;
    double y1 = _face_tracker.top() + _face_tracker.height()/2.0;
    double x2 = _detected_face.x + _detected_face.width/2.0;
    double y2 = _detected_face.y + _detected_face.height/2.0;
    double face_distance = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
 
    return face_distance;
}

void FaceCtrl::FaceTrackersRemove()
{
    for(int i = 0; i < face_trackers_.size(); i ++)
    {
        if(face_trackers_[i].rejected())
        {
            face_trackers_.erase(face_trackers_.begin()+i);
        }
    }
}

void FaceCtrl::FaceTrackersPredict()
{
    //std::cout<< "start prediction" << std::endl;
    if(first_image_)
    {
        cv::Mat raw_image = raw_img_.clone();
        cv::Mat result_image;
        cv::cvtColor(raw_image, result_image, cv::COLOR_BGR2GRAY);
        cv::cvtColor(result_image, result_image, cv::COLOR_GRAY2BGR);

        for(int i = 0; i < face_trackers_.size(); i ++)
        {
            //std::cout << face_trackers_[i].found_frames_ << std::endl;
            if(!face_trackers_[i].accepted())
            {
                continue;
            }
            cv::Rect pred_rect;
            face_trackers_[i].KalmanPredict(pred_rect);
            
            //std::cout << pred_rect << std::endl;
            cv::rectangle(result_image, pred_rect, cv::Scalar(0, 255, 0), 4);
          
            cv::Rect roi_rect = face_trackers_[i].FaceROI();
            //ROI rectangle
            cv::rectangle(result_image, roi_rect, cv::Scalar(0, 0, 255), 1);

            std::vector<cv::Point2f> axes = face_trackers_[i].axes_;
            cv::line(result_image, axes[0], axes[3], cv::Scalar(255,0,0), 5, CV_AA); //blue
            cv::line(result_image, axes[0], axes[2], cv::Scalar(0,255,0), 5, CV_AA); //green
            cv::line(result_image, axes[0], axes[1], cv::Scalar(0,0,255), 5, CV_AA); //red

            for(int j = 0; j < 68; j ++)
            {
                cv::circle(result_image, cv::Point(face_trackers_[i].landmark_.part(j).x(), face_trackers_[i].landmark_.part(j).y()), 2, cv::Scalar(0,255,0));
            }
        }
        cv::Mat result_image_flip;
        //cv::transpose(result_image, result_image_flip);
        //cv::flip(result_image_flip, result_image_flip, 1);
        cv::imshow("InspiRED Face Analysis", result_image);
        cv::waitKey(1);
        sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", result_image).toImageMsg();
        ROS_INFO("Result pub");
        image_pub_.publish(image_msg);
    }
}

void FaceCtrl::FaceDetectInTrackersROI()
{
    //ROS_INFO("Total %d face ROIs", face_trackers_.size());
    if(first_image_)
    {
        for(int i = 0; i < face_trackers_.size(); i ++)
        {
            //ROS_INFO("now detect %dth face's ROI", i);
            cv::Rect face_roi = face_trackers_[i].FaceROI();
            if( (face_roi.width*face_roi.height) < (image_width_*image_height_/5) && (face_roi.width*face_roi.height) > 200)
            {
                //ROS_INFO("Detecting ROI...");
                face_trackers_[i].DetectFaceInROI(& raw_img_);
                //ROS_INFO("Finish detecting ROI..");
            }
        }
    }
}

void FaceCtrl::FacePoseEstimation()
{
    if(first_image_)
    {
        cv::Mat cv_image = raw_img_.clone();
        dlib::cv_image<dlib::bgr_pixel> dlib_image(cv_image);

        for(int i = 0; i < face_trackers_.size(); i ++)
        {
            dlib::rectangle face_rec(face_trackers_[i].left(), face_trackers_[i].top(), face_trackers_[i].left()+face_trackers_[i].width(), face_trackers_[i].top()+face_trackers_[i].height());
            
            double t0 = (double) cv::getTickCount(); 
            face_trackers_[i].landmark_ = face_pose_model_(dlib_image, face_rec);
            estimator_->poseUpdate(face_trackers_[i].landmark_);
            double t1 = (double) cv::getTickCount(); 
            double pose_time = 1000 * (t1 - t0) / cv::getTickFrequency();
            //ROS_INFO("Pose estimation time is: %f ms", pose_time);
            face_trackers_[i].has_landmark_ = true;
            face_trackers_[i].axes_ = estimator_->axes_;
        }
    }
}
/**********************************************************************************************/

int main(int argc, char **argv)
{
    ros::init(argc, argv, "multi_face_tracker");

    cv::namedWindow("Multi-face tracker", 1);
    FaceCtrl faceCtrl;

    ros::Rate r(30);

    while(ros::ok())
    {
        faceCtrl.FaceDetectInTrackersROI();
        faceCtrl.FacePoseEstimation();
        faceCtrl.FaceTrackersPredict();    
        ros::spinOnce();
        faceCtrl.FaceTrackersRemove();
        r.sleep();
    }
}
