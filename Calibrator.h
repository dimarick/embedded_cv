#ifndef EMBEDDED_CV_CALIBRATOR_H
#define EMBEDDED_CV_CALIBRATOR_H

#include <cstdlib>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib.hpp"
#include "opencv2/3d.hpp"

namespace ecv {

    class Calibrator {
    private:
        void convertTo2dPoints(const std::vector<cv::Point3d> &points3d, std::vector<cv::Point2f> &points2d);
        void convertToPlain3dPoints(const std::vector<cv::Point3d> &points1, std::vector<cv::Point3f> &points2);
        void convertTo2dPoints(const std::vector<cv::Point3f> &points3d, std::vector<cv::Point2f> &points2d);
    public:
        double calibrate(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3d>> &objectPoints,
                const std::vector<std::vector<cv::Point3d>> &imagePoints,
                cv::Mat &map1,
                cv::Mat &map2
        );
        double calibrate(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3f>> &objectPoints,
                const std::vector<std::vector<cv::Point3f>> &imagePoints,
                cv::Mat &map1,
                cv::Mat &map2
        );
        double calibrate(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3f>> &objectPoints,
                const std::vector<std::vector<cv::Point2f>> &imagePoints,
                cv::Mat &map1,
                cv::Mat &map2
        );

        void stereoCalibrate(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3f>> &objectPoints,
                const std::vector<std::vector<cv::Point3f>> &imagePoints,
                cv::Mat &map1,
                cv::Mat &map2
        );
    };

} // ecv

#endif //EMBEDDED_CV_CALIBRATOR_H
