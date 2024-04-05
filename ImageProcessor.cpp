#include "ImageProcessor.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

ImageProcessor::ImageProcessor(int capWidth, int capHeight) {
    this->orb = cv::ORB::create(1000, 1.2, 8, 20, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    this->fast = cv::FastFeatureDetector::create();
    this->kaze = cv::KAZE::create();
    this->akaze = cv::AKAZE::create();
    this->mask = cv::UMat::ones(capHeight, capWidth, CV_8UC1);
}

void ImageProcessor::processFrame(cv::UMat image) {
    cv::cvtColor(image, grayImage, cv::COLOR_YUV2GRAY_Y422);
//    cv::normalize(grayImage, grayImage, cv::NORM_HAMMING2);
//    cv::equalizeHist(grayImage, grayImage);
//    kaze->detectAndCompute(grayImage, mask, keyPoints, descriptor);
//    orb->detect(grayImage, keyPoints);
    orb->detectAndCompute(grayImage, mask, keyPoints, descriptor);
//    fast->detect(grayImage, keyPoints, mask);
//    orb->compute(grayImage, keyPoints, descriptor);

    for (auto &keyPoint: keyPoints) {
        cv::circle(image, keyPoint.pt, (int) keyPoint.size, cv::Scalar(255, 255, 255), 1);
    }
}
