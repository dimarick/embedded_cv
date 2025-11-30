#include "ImageProcessor.h"

ImageProcessor::ImageProcessor(int capWidth, int capHeight, mini_server::BroadcastingServer &publisher) : publisher(publisher) {
}

void ImageProcessor::processFrame(cv::UMat &left, cv::UMat &right, cv::UMat &output) {
}
