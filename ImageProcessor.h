#ifndef EMBEDDED_CV_IMAGEPROCESSOR_H
#define EMBEDDED_CV_IMAGEPROCESSOR_H

#include "opencv2/core.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/calib3d.hpp>

class ImageProcessor {
private:
    cv::Ptr<cv::FastFeatureDetector> fast;
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::KAZE> kaze;
    cv::Ptr<cv::AKAZE> akaze;
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
public:
    ImageProcessor(int capWidth, int capHeight);
    void processFrame(cv::UMat &left, cv::UMat &right, cv::UMat &output);
};


#endif //EMBEDDED_CV_IMAGEPROCESSOR_H
