#ifndef EMBEDDED_CV_BLURFRAMEFILTER_H
#define EMBEDDED_CV_BLURFRAMEFILTER_H


#include <core/mat.hpp>

namespace ecv {
    class BlurFrameFilter {
    private:
        const double percentile;
        double currentThreshold = 0.0;
        double currentThresholdStep = 0.05;
        double trueValues = 0;
        double falseValues = 0;
    public:
        explicit BlurFrameFilter(double percentile) : percentile(percentile) {}
        bool apply(const cv::UMat &frame);
    };
} // ecv

#endif //EMBEDDED_CV_BLURFRAMEFILTER_H
