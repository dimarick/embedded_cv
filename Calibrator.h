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
        void convertTo2dPoints(const std::vector<cv::Point3f> &points3d, std::vector<cv::Point2f> &points2d);
    public:
        struct CalibrationData {
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
        };

        double calibrateSingleCamera(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3d>> &objectPoints,
                const std::vector<std::vector<cv::Point3d>> &imagePoints,
                CalibrationData &data
        );

        double calibrateSingleCamera(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3d>> &collectedObjectPoints,
                const std::vector<std::vector<cv::Point3d>> &collectedImagePoints,
                const std::vector<cv::Point3d> &newObjectPoints,
                const std::vector<cv::Point3d> &newImagePoints,
                CalibrationData &data
        );
        double calibrateSingleCamera(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3f>> &objectPoints,
                const std::vector<std::vector<cv::Point3f>> &imagePoints,
                CalibrationData &data
        );
        double calibrateSingleCamera(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3f>> &objectPoints,
                const std::vector<std::vector<cv::Point2f>> &imagePoints,
                CalibrationData &data
        );

        double calibrateCameraPair(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3f>> &objectPointsCam1,
                const std::vector<std::vector<cv::Point2f>> &imagePointsCam1,
                const std::vector<std::vector<cv::Point3f>> &objectPoints,
                const std::vector<std::vector<cv::Point2f>> &imagePoints,
                CalibrationData &dataCam1,
                CalibrationData &data
        );

        double calibrateCameraPair(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3d>> &objectPointsCam1,
                const std::vector<std::vector<cv::Point3d>> &imagePointsCam1,
                const std::vector<std::vector<cv::Point3d>> &objectPoints,
                const std::vector<std::vector<cv::Point3d>> &imagePoints,
                CalibrationData &dataCam1,
                CalibrationData &data
        );
        double calibrateCameraPair(
                cv::Size frameSize,
                const std::set<std::shared_ptr<CalibrateFrameCollector::FramePair>> &pairs,
                const std::vector<cv::Point3d> &objectPointsCam1,
                const std::vector<cv::Point3d> &imagePointsCam1,
                const std::vector<cv::Point3d> &objectPoints,
                const std::vector<cv::Point3d> &imagePoints,
                CalibrationData &dataCam1,
                CalibrationData &data);

        cv::Mat getUndistortMap(cv::Size frameSize, const CalibrationData &data);
        std::pair<cv::Mat, cv::Mat> getUndistortMap(cv::Size frameSize, const CalibrationData &base, const CalibrationData &current);

        double getFx() const {
            return fx;
        }

        double getFy() const {
            return fy;
        }

        void
        printStereoCalibrationStats(const cv::Mat &camMatrixL, const cv::Mat &camMatrixR, const cv::Mat &R,
                                    const cv::Mat &T);
    };
} // ecv

#endif //EMBEDDED_CV_CALIBRATOR_H
