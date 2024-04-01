#include "ImageProcessor.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

ImageProcessor::ImageProcessor(int capWidth, int capHeight) {

    this->orb = cv::ORB::create();
    this->mask = cv::UMat::ones(capHeight, capWidth, CV_8UC1);
}

void ImageProcessor::processFrame(cv::UMat image) {
    orb->detectAndCompute(image, mask, keyPoints, descriptor);

    for (auto &keyPoint: keyPoints) {
        cv::circle(image, keyPoint.pt, (int) keyPoint.size / 5, cv::Scalar(0, 0, 255), 1);
    }
}
