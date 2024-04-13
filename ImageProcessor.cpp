#include "ImageProcessor.h"
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif
#include <opencv2/calib3d.hpp>

ImageProcessor::ImageProcessor(int capWidth, int capHeight) {
    this->orb = cv::ORB::create(300, 2, 4, 5, 0, 2, cv::ORB::FAST_SCORE, 63, 5);
    this->fast = cv::FastFeatureDetector::create();
    this->kaze = cv::KAZE::create();
    this->akaze = cv::AKAZE::create();
    this->mask = cv::UMat::ones(capHeight, capWidth, CV_8UC1);
    this->matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
    this->fmatcher = cv::FlannBasedMatcher();
    this->stereoSGBM = cv::StereoSGBM::create();
    this->stereoBM = cv::StereoBM::create();
    this->stereoBM->setTextureThreshold(0);

    this->q = cv::Mat(4, 4, CV_64F);
    this->q.at<double>(0,0)=1.0;
    this->q.at<double>(0,1)=0.0;
    this->q.at<double>(0,2)=0.0;
    this->q.at<double>(0,3)=-320.0; //cx
    this->q.at<double>(1,0)=0.0;
    this->q.at<double>(1,1)=1.0;
    this->q.at<double>(1,2)=0.0;
    this->q.at<double>(1,3)=-240.0;  //cy
    this->q.at<double>(2,0)=0.0;
    this->q.at<double>(2,1)=0.0;
    this->q.at<double>(2,2)=0.0;
    this->q.at<double>(2,3)=615.0;  //Focal
    this->q.at<double>(3,0)=0.0;
    this->q.at<double>(3,1)=0.0;
    this->q.at<double>(3,2)=1.0/0.10;    //BaseLine
    this->q.at<double>(3,3)= 0.0;  //cx - cx'/baseline
}

void ImageProcessor::processFrame(cv::UMat &left, cv::UMat &right, cv::UMat &output) {
//    static int i = 0;
//    cv::extractChannel(left, grayLeft, 1);
//    cv::extractChannel(right, grayRight, 1);
//
//    grayLeft.convertTo(grayLeft, CV_8UC1);
//    grayRight.convertTo(grayRight, CV_8UC1);
//
//    left.copyTo(grayLeft);
//    right.copyTo(grayRight);

//    auto rectW = (int)(left.cols * 0.01);
//    auto rectH = (int)(left.rows * 0.01);
//    cv::UMat maskRight = cv::UMat::zeros((int)(mask.rows + rectH + 1), mask.cols + rectW + 1, CV_8UC1);
//
//    for (const auto& match : matches) {
//        auto kp1 = keyPointsRight[match.queryIdx];
//        maskRight(cv::Rect((int)kp1.pt.x, std::max(0, (int)(kp1.pt.y - rectH)), (int)rectW, (int)rectH * 2)) = 255;
//    }
//
//    orb->detectAndCompute(grayLeft, mask, keyPointsLeft, descriptorLeft);
//    orb->detectAndCompute(grayRight, mask, keyPointsRight, descriptorRight);

//    cv::drawKeypoints(left, keyPointsLeft, left, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
//    cv::drawKeypoints(left, keyPointsRight, left, cv::Scalar(0, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
//    cv::drawKeypoints(right, keyPointsRight, right, cv::Scalar(0, 255, 255));
//
//    if (keyPointsRight.empty() || keyPointsLeft.empty()) {
//        return;
//    }
//
//    this->stereoBM->compute(grayLeft, grayRight, disparity);
//
//    cv::UMat mindisp;
//    cv::resize(disparity, mindisp, cv::Size(320, 240));
//
//    cv::reprojectImageTo3D(mindisp, scene, q);
//
//    cv::Mat depth;
//
//    cv::extractChannel(scene, depth, 2);
//    depth.convertTo(depth, CV_8UC1);
//    cv::equalizeHist(depth, depth);

//    matcher.match(descriptorRight, descriptorLeft, matches);
//
//    for (const auto& kp1 : keyPointsLeft) {
//        for (const auto& kp2 : keyPointsRight) {
//
//            if (
//                cv::abs(kp1.pt.y - kp2.pt.y) > left.rows * 0.02
//                || cv::abs(kp1.angle - kp2.angle) > 0.5
//                || cv::abs(kp1.size - kp2.size) / kp1.size > 0.5
//                || (kp1.pt.x - kp2.pt.x) > left.cols * 0.02
//                || (kp1.pt.x - kp2.pt.x) < 0
//            ) {
//                continue;
//            }
//
//            cv::circle(left, kp1.pt, 6, cv::Scalar(0, 0, 255));
//            cv::circle(left, kp2.pt, 6, cv::Scalar(0, 255, 255));
//            cv::line(left, kp1.pt, kp2.pt, cv::Scalar(0, 255, 0));
//        }
//    }

//    std::cerr << "Keypoints " << keyPointsLeft.size() << " matches " << matches.size() << std::endl;
}
