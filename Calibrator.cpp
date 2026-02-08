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

        cameraData[0] = 1000.;
        cameraData[4] = 1000.;
        cameraData[2] = (float)frameSize.width / 2;
        cameraData[5] = (float)frameSize.height / 2;
        cameraData[8] = 1.;


        map1.setTo(cv::Scalar::all(0));
        map2.setTo(cv::Scalar::all(0));

        int baseFlags =
//                        cv::CALIB_USE_EXTRINSIC_GUESS|
                cv::CALIB_USE_LU|
                cv::CALIB_RATIONAL_MODEL|
                cv::CALIB_TILTED_MODEL|
                cv::CALIB_THIN_PRISM_MODEL|
                cv::CALIB_FIX_ASPECT_RATIO|
//                cv::CALIB_FIX_PRINCIPAL_POINT|
                cv::CALIB_USE_INTRINSIC_GUESS;

        double result = 0;

        try {

            result = cv::calibrateCamera(objectPointsSlice, imagePointsSlice, frameSize, cameraMatrix,
                                distCoeff, rvecs, tvecs, baseFlags,
                                cv::TermCriteria(10, 1e-7)
            );

        } catch (const std::exception &e) {
            std::cerr << "cv::calibrateCamera failed" << std::endl;

            return false;
        }

        std::cout << std::format("result = {}, camera.fx = {}, camera.fy = {}, camera.cx = {}, camera.cy = {}, distCoeff = {}\n",
                                 result, cameraData[0], cameraData[4], cameraData[2], cameraData[5], distCoeff);

        cv::initUndistortRectifyMap(cameraMatrix, distCoeff, cv::noArray(), cameraMatrix, frameSize,
                                    CV_32FC2,
                                    map1, map2);

        return true;
    }

    void Calibrator::convertTo2dPoints(const std::vector<cv::Point3d> &points3d, std::vector<cv::Point2f> &points2d) {
        for (int j = 0; j < points2d.size(); ++j) {
            points2d[j] = cv::Point(points3d[j].x, points3d[j].y);
        }
    }

    void Calibrator::convertToPlain3dPoints(const std::vector<cv::Point3d> &points1, std::vector<cv::Point3f> &points2) {
        for (int j = 0; j < points2.size(); ++j) {
            points2[j] = cv::Point3f(points1[j].x, points1[j].y, 0);
        }
    }

    void Calibrator::convertTo2dPoints(const std::vector<cv::Point3f> &points3d, std::vector<cv::Point2f> &points2d) {
        for (int j = 0; j < points2d.size(); ++j) {
            points2d[j] = cv::Point(points3d[j].x, points3d[j].y);
        }
    }

    void Calibrator::convertToPlain3dPoints(const std::vector<cv::Point3f> &points1, std::vector<cv::Point3f> &points2) {
        for (int j = 0; j < points2.size(); ++j) {
            points2[j] = cv::Point3f(points1[j].x, points1[j].y, 0);
        }
    }

    bool Calibrator::calibrate(cv::Size frameSize, const std::vector<std::vector<cv::Point3d>> &objectPoints,
                               const std::vector<std::vector<cv::Point3d>> &imagePoints, cv::Mat &map1, cv::Mat &map2) {
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

        return calibrate(frameSize, objectPoints2, imagePoints2, map1, map2);
    }

    bool Calibrator::calibrate(cv::Size frameSize, const std::vector<std::vector<cv::Point3f>> &objectPoints,
                               const std::vector<std::vector<cv::Point3f>> &imagePoints, cv::Mat &map1, cv::Mat &map2) {
        std::vector<std::vector<cv::Point2f>> imagePoints2;
        for (int i = 0; i < imagePoints.size(); ++i) {
            imagePoints2[i].resize(imagePoints[i].size());
            convertTo2dPoints(imagePoints[i], imagePoints2[i]);
        }

        return calibrate(frameSize, objectPoints, imagePoints2, map1, map2);
    }
} // ecv