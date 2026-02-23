#ifndef EMBEDDED_CV_CALIBRATOR_H
#define EMBEDDED_CV_CALIBRATOR_H

#include <cstdlib>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib.hpp"
#include "opencv2/3d.hpp"
#include "CalibrateFrameCollector.h"

namespace ecv {

    class Calibrator {
    private:
        double fx = 700.; // значения в диапазоне 100-10000 довольно быстро сходятся к ожидаемому
        double fy = 700.; // значения в диапазоне 100-10000 довольно быстро сходятся к ожидаемому
        double cx = 0.;
        double cy = 0.;

        void convertTo2dPoints(const std::vector<cv::Point3d> &points3d, std::vector<cv::Point2f> &points2d);
        void convertToPlain3dPoints(const std::vector<cv::Point3d> &points1, std::vector<cv::Point3f> &points2);
    public:
        struct CalibrationData {
            int callCount = 0;
            cv::Mat cameraMatrix;
            std::vector<double> distCoeff;
            cv::Mat rvecs;
            cv::Mat tvecs;
            cv::Mat R;
            cv::Mat T;
            cv::Mat E;
            cv::Mat F;

            CalibrationData() {
                cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);
                rvecs = cv::Mat::zeros(3, 3, CV_64F);
                tvecs = cv::Mat::zeros(3, 3, CV_64F);
                distCoeff = std::vector<double>(12);
            }

            CalibrationData(const CalibrationData& other) {
                cameraMatrix = other.cameraMatrix.clone();
                distCoeff = other.distCoeff;
                rvecs = other.rvecs.clone();
                tvecs = other.tvecs.clone();
                R = other.R.clone();
                T = other.T.clone();
                E = other.E.clone();
                F = other.F.clone();
            }

            CalibrationData& operator=(const CalibrationData& other) {
                if (this != &other) {
                    cameraMatrix = other.cameraMatrix.clone();
                    distCoeff = other.distCoeff;               // std::vector копируется глубоко
                    rvecs = other.rvecs.clone();
                    tvecs = other.tvecs.clone();
                    R = other.R.clone();
                    T = other.T.clone();
                    E = other.E.clone();
                    F = other.F.clone();
                }
                return *this;
            }
        };

        double calibrateSingleCamera(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3d>> &objectPoints,
                const std::vector<std::vector<cv::Point3d>> &imagePoints,
                CalibrationData &data,
                int flags,
                cv::TermCriteria term = cv::TermCriteria(10, 1e-7)
        );
        double calibrateSingleCamera(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3f>> &objectPoints,
                const std::vector<std::vector<cv::Point2f>> &imagePoints,
                CalibrationData &data,
                int flags,
                cv::TermCriteria term = cv::TermCriteria(10, 1e-7)
        );

        double validateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3d>> &objectPoints,
                                    const std::vector<std::vector<cv::Point3d>> &imagePoints, const CalibrationData &data);
    };
} // ecv

#endif //EMBEDDED_CV_CALIBRATOR_H
