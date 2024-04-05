#ifndef EMBEDDED_CV_IMAGEPROCESSOR_H
#define EMBEDDED_CV_IMAGEPROCESSOR_H

#include "opencv2/core.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/core/ocl.hpp>

class ImageProcessor {
private:
    cv::Ptr<cv::FastFeatureDetector> fast;
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::KAZE> kaze;
    cv::Ptr<cv::AKAZE> akaze;
    std::vector<cv::KeyPoint> keyPoints;
    cv::UMat mask;
    cv::UMat descriptor;
    cv::UMat grayImage;
public:
    ImageProcessor(int capWidth, int capHeight);
    void processFrame(cv::UMat image);
};


#endif //EMBEDDED_CV_IMAGEPROCESSOR_H
