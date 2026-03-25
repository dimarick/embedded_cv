// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

#include "Calibrator.h"
#include <opencv2/calib3d.hpp>
#include <random>

namespace ecv {
    double Calibrator::calibrateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3f>> &objectPoints,
                                             const std::vector<std::vector<cv::Point2f>> &imagePoints,
                                             CalibrationData &data,
                                             const std::vector<double> &stdDeviationsIntrinsics,
                                             const std::vector<double> &perViewErrors,
                                             int flags,
                                             cv::TermCriteria term) const {
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

        int baseFlags =
                cv::CALIB_USE_LU|
                cv::CALIB_RATIONAL_MODEL|
                cv::CALIB_TILTED_MODEL|
                cv::CALIB_THIN_PRISM_MODEL|
                cv::CALIB_FIX_ASPECT_RATIO;

        if (data.frameCount > 0) {
            baseFlags |= cv::CALIB_USE_INTRINSIC_GUESS;
        }
//
//        if (data.frameCount > 10) {
//            baseFlags |= cv::CALIB_RATIONAL_MODEL;
//        }
//
//        if (data.frameCount > 20) {
//            baseFlags |= cv::CALIB_TILTED_MODEL;
//        }
//
//        if (data.frameCount > 30) {
//            baseFlags |= cv::CALIB_THIN_PRISM_MODEL;
//        }

        double result = 0;

        try {
            cv::Mat stdDeviationsExtrinsics;
            result = cv::calibrateCamera(objectPointsSlice, imagePointsSlice, frameSize, data.cameraMatrix,
                                         data.distCoeff, data.rvecs, data.tvecs, stdDeviationsIntrinsics,
                                         stdDeviationsExtrinsics, perViewErrors, baseFlags | flags, term);
        } catch (const std::exception &e) {
            std::cerr << "cv::calibrateCamera failed: " << e.what() << std::endl;

            return 1. / 0.;
        }

        data.frameCount += inputSize;

        return result;
    }

    template <typename T> void Calibrator::mergeCalibrationItem(const T &newData, double newDataDev, T &existsData, double existsDataDev) const {
        if (existsDataDev == 0) {
            existsData *= (1 - newDataDev);
            existsData += newDataDev * newData;
            return;
        }

        std::normal_distribution<double> generalization(1., existsDataDev);
        std::mt19937 r {std::random_device{}()};

        double randomValue = generalization(r);

        // Чтобы объекты обрабатывались in-place
        existsData *= randomValue * (1 - newDataDev);
        existsData += randomValue * newDataDev * newData;
    }

    void Calibrator::mergeIntrinsics(const CalibrationData &newData, double newDataDev, CalibrationData &existsData, double existsDataDev) const {
        mergeCalibrationItem(std::clamp(newData.cameraMatrix.at<double>(0, 0), 0., 10000.), newDataDev, existsData.cameraMatrix.at<double>(0, 0), existsDataDev);
        existsData.cameraMatrix.at<double>(1, 1) = existsData.cameraMatrix.at<double>(0, 0);
        mergeCalibrationItem(std::clamp(newData.cameraMatrix.at<double>(0, 2), 0., 3000.), newDataDev, existsData.cameraMatrix.at<double>(0, 2), existsDataDev);
        mergeCalibrationItem(std::clamp(newData.cameraMatrix.at<double>(1, 2), 0., 3000.), newDataDev, existsData.cameraMatrix.at<double>(1, 2), existsDataDev);

        for (int i = 0; i < std::min(newData.distCoeff.size(), existsData.distCoeff.size()); ++i) {
            mergeCalibrationItem(std::clamp(newData.distCoeff[i], -1e2, 1e2), newDataDev, existsData.distCoeff[i], existsDataDev);
        }
        existsData.rvecs = newData.rvecs;
        existsData.tvecs = newData.tvecs;
        existsData.frameCount = newData.frameCount;
    }

    void Calibrator::convertTo2dPoints(const std::vector<cv::Point3d> &points3d, std::vector<cv::Point2f> &points2d) const {
        for (int j = 0; j < points2d.size(); ++j) {
            points2d[j] = cv::Point2f((float)points3d[j].x, (float)points3d[j].y);
        }
    }

    void Calibrator::convertToPlain3dPoints(const std::vector<cv::Point3d> &points1, std::vector<cv::Point3f> &points2) const {
        for (int j = 0; j < points2.size(); ++j) {
            points2[j] = cv::Point3f((float)points1[j].x, (float)points1[j].y, 0);
        }
    }

    double Calibrator::calibrateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3d>> &objectPoints,
                                             const std::vector<std::vector<cv::Point3d>> &imagePoints,
                                             CalibrationData &data,
                                             const std::vector<double> &stdDeviationsIntrinsics,
                                             const std::vector<double> &perViewErrors,
                                             int flags,
                                             cv::TermCriteria term) const {
        std::vector<std::vector<cv::Point3f>> objectPoints2(objectPoints.size());
        std::vector<std::vector<cv::Point2f>> imagePoints2(imagePoints.size());

        for (int i = 0; i < imagePoints.size(); ++i) {
            imagePoints2[i].resize(imagePoints[i].size());
            convertTo2dPoints(imagePoints[i], imagePoints2[i]);
        }

        for (int i = 0; i < objectPoints.size(); ++i) {
            objectPoints2[i].resize(objectPoints[i].size());
            convertToPlain3dPoints(objectPoints[i], objectPoints2[i]);
        }

        return calibrateSingleCamera(frameSize, objectPoints2, imagePoints2, data, stdDeviationsIntrinsics, perViewErrors, flags, term);
    }

    double Calibrator::validateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3d>> &objectPoints,
                                             const std::vector<std::vector<cv::Point3d>> &imagePoints,
                                             const CalibrationData &data) const {
        int flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_TILTED_MODEL |
                    cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_FIX_PRINCIPAL_POINT |
                    cv::CALIB_FIX_FOCAL_LENGTH | cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 |
                    cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6 | cv::CALIB_FIX_S1_S2_S3_S4 |
                    cv::CALIB_FIX_TAUX_TAUY | cv::CALIB_FIX_TANGENT_DIST | cv::CALIB_FIX_INTRINSIC |
                    cv::CALIB_FIX_SKEW;

        auto testData = data;

        return calibrateSingleCamera(frameSize, objectPoints, imagePoints, testData, std::vector<double>(CALIB_NINTRINSIC), std::vector<double>(objectPoints.size()), flags, cv::TermCriteria(1, 1e-7));
    }
} // ecv