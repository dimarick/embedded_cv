#ifndef EMBEDDED_CV_CALIBRATOR_H
#define EMBEDDED_CV_CALIBRATOR_H

#include <cstdlib>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib.hpp"
#include "opencv2/3d.hpp"

namespace ecv {

    class Calibrator {
    public:
        bool calibrate(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3f>> &objectPoints,
                const std::vector<std::vector<cv::Point2f>> &imagePoints,
                cv::Mat &map1,
                cv::Mat &map2
        );
    };

} // ecv

#endif //EMBEDDED_CV_CALIBRATOR_H
