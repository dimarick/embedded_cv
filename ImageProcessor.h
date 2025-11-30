#ifndef EMBEDDED_CV_IMAGEPROCESSOR_H
#define EMBEDDED_CV_IMAGEPROCESSOR_H

#include "opencv2/core.hpp"
#include <BroadcastingServer.h>

class ImageProcessor {
private:
    mini_server::BroadcastingServer &publisher;
    std::vector<cv::DMatch> matches;

    std::vector<cv::UMat> recentFrames;
    int currentFrame = 0;
public:
    std::atomic<float> denoiseLevel = 3;
    ImageProcessor(int capWidth, int capHeight, mini_server::BroadcastingServer &publisher);
    void processFrame(cv::UMat &left, cv::UMat &right, cv::UMat &output);
};


#endif //EMBEDDED_CV_IMAGEPROCESSOR_H
