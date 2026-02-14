#include <imgproc.hpp>
#include "BlurFrameFilter.h"

using namespace ecv;

bool BlurFrameFilter::apply(const cv::UMat &frame) {
    cv::UMat thumb, lap;
    cv::extractChannel(frame, thumb, 1); // green
    cv::resize(frame, thumb, cv::Size(320, 240), cv::INTER_NEAREST);
    cv::Laplacian(thumb, lap, CV_32F);
    cv::absdiff(lap, 0, lap);
    auto mean = cv::mean(lap)[0];

    if (currentThreshold == 0) {
        currentThreshold = mean;
    }

    bool solution = mean > currentThreshold;
    if (solution) {
        trueValues++;
    } else {
        falseValues++;
    }

    if (trueValues / falseValues > percentile) {
        currentThreshold *= 1 + currentThresholdStep;
    } else {
        currentThreshold /= 1 + currentThresholdStep;
    }

    if ((trueValues + falseValues) > 1000) {
        trueValues *= 0.95;
        falseValues *= 0.95;
    }

    return solution;
}
