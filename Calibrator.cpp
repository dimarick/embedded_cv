//
// Created by dima on 25.10.25.
//

#include "Calibrator.h"

namespace ecv {
    bool Calibrator::calibrate(cv::Size frameSize, const std::vector<std::vector<cv::Point3f>> &objectPoints,
                               const std::vector<std::vector<cv::Point2f>> &imagePoints,
                               cv::Mat &map1, cv::Mat &map2) {
        size_t objectSize = 0;
        for (const auto & objectPoint : objectPoints) {
            if (objectPoint.empty()) {
                break;
            }
            objectSize++;
        }

        size_t imageSize = 0;
        for (const auto & imagePoint : imagePoints) {
            if (imagePoint.empty()) {
                break;
            }
            imageSize++;
        }

        auto inputSize = std::min(objectSize, imageSize);

        auto objectPointsSlice = objectPoints;
        auto imagePointsSlice = imagePoints;
        if (inputSize < imagePoints.size()) {
            objectPointsSlice = std::vector(objectPoints.begin(), objectPoints.begin() + (int)inputSize);
            imagePointsSlice = std::vector(imagePoints.begin(), imagePoints.begin() + (int)inputSize);
        }

        cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);
        std::vector<double> distCoeff(12);
        cv::Mat rvecs = cv::Mat::zeros(3, 3, CV_64F);
        cv::Mat tvecs = cv::Mat::zeros(3, 3, CV_64F);

        cameraMatrix.setTo(cv::Scalar::all(0));
        auto cameraData = (double *)cameraMatrix.data;

        cameraData[0] = 8000.;
        cameraData[4] = 8000.;
        cameraData[8] = 1.;


        map1.setTo(cv::Scalar::all(0));
        map2.setTo(cv::Scalar::all(0));

        int baseFlags =
//                        cv::CALIB_USE_EXTRINSIC_GUESS|
                cv::CALIB_FIX_PRINCIPAL_POINT|
                cv::CALIB_RATIONAL_MODEL|
                cv::CALIB_THIN_PRISM_MODEL;

        std::vector<int> flagsChain = {
//                        cv::CALIB_USE_INTRINSIC_GUESS|
//                        cv::CALIB_FIX_FOCAL_LENGTH|
                cv::CALIB_USE_LU|
                cv::CALIB_FIX_TAUX_TAUY|
                cv::CALIB_FIX_TANGENT_DIST|
                cv::CALIB_FIX_S1_S2_S3_S4|
                //                        cv::CALIB_FIX_K1|
                //                        cv::CALIB_FIX_K2|
                //                        cv::CALIB_FIX_K3|
                //                        cv::CALIB_FIX_K4|
                cv::CALIB_FIX_K5|
                cv::CALIB_FIX_K6|
                cv::CALIB_FIX_ASPECT_RATIO,

                cv::CALIB_USE_INTRINSIC_GUESS|
                cv::CALIB_FIX_FOCAL_LENGTH|
                cv::CALIB_FIX_TAUX_TAUY|
                cv::CALIB_FIX_TANGENT_DIST|
                //                        cv::CALIB_FIX_S1_S2_S3_S4|
                //                        cv::CALIB_FIX_K1|
                //                        cv::CALIB_FIX_K2|
                //                        cv::CALIB_FIX_K3|
                //                        cv::CALIB_FIX_K4|
                //                        cv::CALIB_FIX_K5|
                //                        cv::CALIB_FIX_K6|
                cv::CALIB_FIX_ASPECT_RATIO,

                cv::CALIB_USE_INTRINSIC_GUESS|
                //                        cv::CALIB_FIX_FOCAL_LENGTH|
                cv::CALIB_FIX_TAUX_TAUY|
                //                        cv::CALIB_FIX_TANGENT_DIST|
                //                        cv::CALIB_FIX_S1_S2_S3_S4|
                cv::CALIB_FIX_K1|
                cv::CALIB_FIX_K2|
                cv::CALIB_FIX_K3|
                cv::CALIB_FIX_K4|
                cv::CALIB_FIX_K5|
                cv::CALIB_FIX_K6|
                cv::CALIB_FIX_ASPECT_RATIO,
        };
        try {

            cv::calibrateCamera(objectPointsSlice, imagePointsSlice, frameSize, cameraMatrix,
                                distCoeff, rvecs, tvecs, baseFlags | flagsChain[0],
                                cv::TermCriteria(10, 0.0001)
            );

            for (auto i = 1; i < flagsChain.size(); i++) {
                cv::calibrateCamera(objectPointsSlice, imagePointsSlice, frameSize, cameraMatrix,
                                    distCoeff, rvecs, tvecs, baseFlags | flagsChain[i],
                                    cv::TermCriteria(1000, 1e-7)
                );
            }
        } catch (const std::exception &e) {
            std::cerr << "cv::calibrateCamera failed" << std::endl;

            return false;
        }

        cv::initUndistortRectifyMap(cameraMatrix, distCoeff, cv::noArray(), cameraMatrix, frameSize,
                                    CV_32FC2,
                                    map1, map2);

        return true;
    }
} // ecv