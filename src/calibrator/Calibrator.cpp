#include "Calibrator.h"
#include <opencv2/calib3d.hpp>
#include <iomanip>

namespace ecv {
    double Calibrator::calibrateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3f>> &objectPoints,
                                             const std::vector<std::vector<cv::Point2f>> &imagePoints,
                                             CalibrationData &data,
                                             int flags,
                                             cv::TermCriteria term) {
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

        auto cameraData = (double *)data.cameraMatrix.data;

        fx = cameraData[0] == 0 ? 1000 : cameraData[0];
        fy = cameraData[4] == 0 ? 1000 : cameraData[4];

        cx = cameraData[2] == 0 ? (float)frameSize.width / 2 : cameraData[2];
        cy = cameraData[5] == 0 ? (float)frameSize.height / 2 : cameraData[5];

        cameraData[0] = fx;
        cameraData[4] = fy;
        cameraData[2] = cx;
        cameraData[5] = cy;
        cameraData[8] = 1.;

        int baseFlags =
                cv::CALIB_USE_LU|
                cv::CALIB_RATIONAL_MODEL|
                cv::CALIB_TILTED_MODEL|
                cv::CALIB_THIN_PRISM_MODEL|
                cv::CALIB_FIX_ASPECT_RATIO;

        if (data.callCount > 0) {
            baseFlags |= cv::CALIB_USE_INTRINSIC_GUESS;
        }


        double result = 0;

        std::vector<double> distCoeff = data.distCoeff;

        try {
            result = cv::calibrateCamera(objectPointsSlice, imagePointsSlice, frameSize, data.cameraMatrix,
                                data.distCoeff, data.rvecs, data.tvecs, baseFlags | flags, term);
        } catch (const std::exception &e) {
            std::cerr << "cv::calibrateCamera failed" << std::endl;

            return 1. / 0.;
        }

        double ema = data.callCount > 0 ? 2. / (result * 20 + 1.) : 1;

        if ((flags & cv::CALIB_FIX_INTRINSIC) == 0) {
            fx = ema * cameraData[0] + (1 - ema) * fx;
            fy = ema * cameraData[4] + (1 - ema) * fy;
            cx = ema * cameraData[2] + (1 - ema) * cx;
            cy = ema * cameraData[5] + (1 - ema) * cy;
        }

        for (int i = 0; i < distCoeff.size(); ++i) {
            data.distCoeff[i] = ema * data.distCoeff[i] + (1 - ema) * distCoeff[i];
        }

        data.callCount++;

        return result;
    }

    void Calibrator::convertTo2dPoints(const std::vector<cv::Point3d> &points3d, std::vector<cv::Point2f> &points2d) {
        for (int j = 0; j < points2d.size(); ++j) {
            points2d[j] = cv::Point2f(points3d[j].x, points3d[j].y);
        }
    }

    void Calibrator::convertToPlain3dPoints(const std::vector<cv::Point3d> &points1, std::vector<cv::Point3f> &points2) {
        for (int j = 0; j < points2.size(); ++j) {
            points2[j] = cv::Point3f(points1[j].x, points1[j].y, 0);
        }
    }

    double Calibrator::calibrateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3d>> &objectPoints,
                                             const std::vector<std::vector<cv::Point3d>> &imagePoints,
                                             CalibrationData &data,
                                             int flags,
                                             cv::TermCriteria term) {
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

        return calibrateSingleCamera(frameSize, objectPoints2, imagePoints2, data, flags, term);
    }

    double Calibrator::validateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3d>> &objectPoints,
                                             const std::vector<std::vector<cv::Point3d>> &imagePoints,
                                             const CalibrationData &data) {
        int flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_TILTED_MODEL |
                    cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_FIX_PRINCIPAL_POINT |
                    cv::CALIB_FIX_FOCAL_LENGTH | cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 |
                    cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6 | cv::CALIB_FIX_S1_S2_S3_S4 |
                    cv::CALIB_FIX_TAUX_TAUY | cv::CALIB_FIX_TANGENT_DIST | cv::CALIB_FIX_INTRINSIC |
                    cv::CALIB_FIX_SKEW;

        auto testData = data;

        return calibrateSingleCamera(frameSize, objectPoints, imagePoints, testData, flags, cv::TermCriteria(1, 1e-7));
    }
} // ecv