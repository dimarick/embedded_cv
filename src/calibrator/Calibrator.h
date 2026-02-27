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

        template<typename T> void mergeCalibrationItem(const T &newData, double newDataDev, T &existsData, double existsDataDev);
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
            cv::Mat Ts;
            cv::Mat Rs;
            cv::Mat Ri;
            cv::Mat Pi;

            CalibrationData() {
                cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);
                rvecs = cv::Mat::zeros(3, 3, CV_64F);
                tvecs = cv::Mat::zeros(3, 3, CV_64F);
                distCoeff = std::vector<double>(14);
                cameraMatrix.at<double>(0, 0) = 1000.;
                cameraMatrix.at<double>(1, 1) = 1000.;
                cameraMatrix.at<double>(2, 2) = 1;
            }

            explicit CalibrationData(const cv::Size &frameSize) : CalibrationData() {
                cameraMatrix.at<double>(0, 2) = (double)frameSize.width / 2.;
                cameraMatrix.at<double>(1, 2) = (double)frameSize.height / 2.;
            }

            void copyTo(CalibrationData& target) const {
                target.callCount = callCount;
                target.cameraMatrix = cameraMatrix.clone();
                target.distCoeff = distCoeff;
                target.rvecs = rvecs.clone();
                target.tvecs = tvecs.clone();
                target.R = R.clone();
                target.T = T.clone();
                target.E = E.clone();
                target.F = F.clone();
                target.Rs = Rs.clone();
                target.Ts = Ts.clone();
                target.Ri = Ri.clone();
                target.Pi = Pi.clone();
            }

            CalibrationData(const CalibrationData& other) {
                other.copyTo(*this);
            }

            CalibrationData& operator=(const CalibrationData& other) {
                if (this != &other) {
                    other.copyTo(*this);
                }
                return *this;
            }

            void store(cv::FileStorage &storage) const {
                storage << "data" << "{";
                storage << "cameraMatrix" << cameraMatrix;
                storage << "distCoeff" << distCoeff;
                storage << "R" << R;
                storage << "T" << T;
                storage << "Rs" << Rs;
                storage << "Ts" << Ts;
                storage << "Ri" << Ri;
                storage << "Pi" << Pi;
                storage << "}";
            }
        };

        double calibrateSingleCamera(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3d>> &objectPoints,
                const std::vector<std::vector<cv::Point3d>> &imagePoints,
                CalibrationData &data,
                const std::vector<double> &stdDeviationsIntrinsics,
                const std::vector<double> &perViewErrors,
                int flags,
                cv::TermCriteria term = cv::TermCriteria(10, 1e-7)
        );
        double calibrateSingleCamera(
                cv::Size frameSize,
                const std::vector<std::vector<cv::Point3f>> &objectPoints,
                const std::vector<std::vector<cv::Point2f>> &imagePoints,
                CalibrationData &data,
                const std::vector<double> &stdDeviationsIntrinsics,
                const std::vector<double> &perViewErrors,
                int flags,
                cv::TermCriteria term = cv::TermCriteria(10, 1e-7)
        );

        double validateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3d>> &objectPoints,
                                    const std::vector<std::vector<cv::Point3d>> &imagePoints, const CalibrationData &data);

        void mergeIntrinsics(const CalibrationData &newData, double newDataDev, CalibrationData &existsData,
                             double existsDataDev);
    };
} // ecv

#endif //EMBEDDED_CV_CALIBRATOR_H
