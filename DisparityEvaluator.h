//
// Created by dima on 03.11.25.
//

#ifndef EMBEDDED_CV_DISPARITYEVALUATOR_H
#define EMBEDDED_CV_DISPARITYEVALUATOR_H

#include <vector>
#include "opencv2/core.hpp"

namespace ecv {

    class DisparityEvaluator {
    public:
        void evaluateDisparity(const std::vector<cv::Mat> &frames, cv::Mat &disparity) const;
        void evaluateIncrementally(const std::vector<cv::Mat> &frames, const cv::Mat &roughDisparity, cv::Mat &disparity) const;
    };

} // ecv

#endif //EMBEDDED_CV_DISPARITYEVALUATOR_H
