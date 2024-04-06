#include "ImageProcessor.h"
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif

ImageProcessor::ImageProcessor(int capWidth, int capHeight) {
    this->orb = cv::ORB::create(300, 2, 4, 5, 0, 2, cv::ORB::FAST_SCORE, 63, 5);
    this->fast = cv::FastFeatureDetector::create();
    this->kaze = cv::KAZE::create();
    this->akaze = cv::AKAZE::create();
    this->mask = cv::UMat::ones(capHeight, capWidth, CV_8UC1);
    this->matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
    this->fmatcher = cv::FlannBasedMatcher();
}

void ImageProcessor::processFrame(cv::UMat &left, cv::UMat &right, cv::UMat &output) {
    static int i = 0;
//    cv::extractChannel(left, grayLeft, 1);
//    cv::extractChannel(right, grayRight, 1);

    left.copyTo(grayLeft);
    right.copyTo(grayRight);

    orb->detectAndCompute(grayLeft, mask, keyPointsLeft, descriptorLeft);
    orb->detectAndCompute(grayRight, mask, keyPointsRight, descriptorRight);

//    cv::drawKeypoints(left, keyPointsLeft, left, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
//    cv::drawKeypoints(left, keyPointsRight, left, cv::Scalar(0, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
//    cv::drawKeypoints(right, keyPointsRight, right, cv::Scalar(0, 255, 255));

    if (keyPointsRight.empty() || keyPointsLeft.empty()) {
        return;
    }

    matcher.match(descriptorRight, descriptorLeft, matches);

//    cv::drawMatches(left, keyPointsLeft, right, keyPointsRight, matches, output, cv::Scalar(0, 255), cv::Scalar(0, 0, 255));

    std::cerr << "Keypoints " << keyPointsLeft.size() << " matches " << matches.size() << std::endl;
}
