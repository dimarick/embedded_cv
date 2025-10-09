#ifndef EMBEDDED_CV_IMAGEPROCESSOR_H
#define EMBEDDED_CV_IMAGEPROCESSOR_H

#include "opencv2/core.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/calib3d.hpp>
#include <BroadcastingServer.h>

class ImageProcessor {
private:
    mini_server::BroadcastingServer &publisher;
    cv::Ptr<cv::FastFeatureDetector> fast;
    cv::Ptr<cv::ORB> orb;
    std::vector<cv::KeyPoint> keyPointsLeft;
    std::vector<cv::KeyPoint> keyPointsRight;
    cv::UMat mask;
    cv::UMat descriptorLeft;
    cv::UMat descriptorRight;
    cv::UMat grayLeft;
    cv::UMat grayRight;
    cv::UMat disparity;
    cv::UMat scene;
    cv::Mat q;
    cv::BFMatcher matcher;
    cv::FlannBasedMatcher fmatcher;
    std::vector<cv::DMatch> matches;

    cv::Ptr<cv::StereoSGBM> stereoSGBM;
    cv::Ptr<cv::StereoBM> stereoBM;
    std::vector<cv::UMat> recentFrames;
    int currentFrame = 0;
public:
    std::atomic<float> denoiseLevel = 3;
    ImageProcessor(int capWidth, int capHeight, mini_server::BroadcastingServer &publisher);
    void processFrame(cv::UMat &left, cv::UMat &right, cv::UMat &output);
};


#endif //EMBEDDED_CV_IMAGEPROCESSOR_H
