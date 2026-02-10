//
// Created by dima on 25.10.25.
//

#include "Calibrator.h"

namespace ecv {
    double Calibrator::calibrate(cv::Size frameSize, const std::vector<std::vector<cv::Point3f>> &objectPoints,
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

        cx = cx == 0 ? (float)frameSize.width / 2 : cx;
        cy = cy == 0 ? (float)frameSize.height / 2 : cy;

        cameraData[0] = fx;
        cameraData[4] = fy;
        cameraData[2] = cx;
        cameraData[5] = cy;
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
                cv::CALIB_USE_INTRINSIC_GUESS;

        double result = 0;

        try {

            result = cv::calibrateCamera(objectPointsSlice, imagePointsSlice, frameSize, cameraMatrix,
                                distCoeff, rvecs, tvecs, baseFlags,
                                cv::TermCriteria(10, 1e-7)
            );

        } catch (const std::exception &e) {
            std::cerr << "cv::calibrateCamera failed" << std::endl;

            return 1. / 0.;
        }

        std::cout << std::format("result = {}, camera.fx = {}, camera.fy = {}, camera.cx = {}, camera.cy = {}, distCoeff = {}\n",
                                 result, cameraData[0], cameraData[4], cameraData[2], cameraData[5], distCoeff);

        auto ema = 2. / (5. + 1.);
        fx = ema * cameraData[0] + (1 - ema) * fx;
        fy = ema * cameraData[4] + (1 - ema) * fy;
        cx = ema * cameraData[2] + (1 - ema) * cx;
        cy = ema * cameraData[5] + (1 - ema) * cy;

        cv::initUndistortRectifyMap(cameraMatrix, distCoeff, cv::noArray(), cameraMatrix, frameSize,
                                    CV_32FC2,
                                    map1, map2);

        return result;
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

    double Calibrator::calibrate(cv::Size frameSize, const std::vector<std::vector<cv::Point3d>> &objectPoints,
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

    double Calibrator::calibrate(cv::Size frameSize, const std::vector<std::vector<cv::Point3f>> &objectPoints,
                               const std::vector<std::vector<cv::Point3f>> &imagePoints, cv::Mat &map1, cv::Mat &map2) {
        std::vector<std::vector<cv::Point2f>> imagePoints2;
        for (int i = 0; i < imagePoints.size(); ++i) {
            imagePoints2[i].resize(imagePoints[i].size());
            convertTo2dPoints(imagePoints[i], imagePoints2[i]);
        }

        return calibrate(frameSize, objectPoints, imagePoints2, map1, map2);
    }

    void Calibrator::stereoCalibrate(cv::Size frameSize, const std::vector<std::vector<cv::Point3f>> &objectPoints,
                                     const std::vector<std::vector<cv::Point3f>> &imagePoints, cv::Mat &map1,
                                     cv::Mat &map2) {
//        cv::stereoCalibrate();


//        if (plainWidth[0] == plainWidth[1] && plainHeight[0] == plainHeight[1] && plainHeight[0] > 0 && plainHeight[1] > 0) {
//            const auto &baseFrame = plain[0];
//            const auto &otherFrame = plain[1];
//
//            const auto &baseSize = baseFrame.size();
//            cv::Point2f imageCenter = {(float)(baseSize.width - 1) / 2, (float)(baseSize.height - 1) / 2};
//            auto gridCenter = findNearestPoint(imageCenter, plainImageGrid[0], (double)calibrateMapper3[0].patternSize);
//            auto gridCenter2 = findNearestPoint(imageCenter, plainImageGrid[1], (double)calibrateMapper3[1].patternSize);
//
//            if (gridCenter < 0 || gridCenter2 < 0) {
//                continue;
//            }
//
//            cv::Mat transform;
//
//            auto gridCenterX = (size_t) gridCenter % plainWidth[0];
//            auto gridCenterY = (size_t) gridCenter / plainWidth[0];
//            auto gridCenterX2 = (size_t) gridCenter2 % plainWidth[1];
//            auto gridCenterY2 = gridCenterY;
//            auto rw = std::min({
//                                       gridCenterX,
//                                       gridCenterX2,
//                                       plainWidth[0] - gridCenterX,
//                                       plainWidth[1] - gridCenterX2,
//                               });
//            auto rh = std::min({
//                                       gridCenterY,
//                                       gridCenterY2,
//                                       plainHeight[0] - gridCenterY,
//                                       plainHeight[1] - gridCenterY2,
//                               });
//
//            if (rw < 1) {
//                continue;
//            }
//
//            if (rh < 1) {
//                continue;
//            }
//
//            const std::vector<cv::Point2f> src = {
//                    {(float)plainImageGrid[0][(gridCenterY - rh) * plainWidth[0] + gridCenterX - rw].x, (float)plainImageGrid[0][(gridCenterY - rh) * plainWidth[0] + gridCenterX - rw].y},
//                    {(float)plainImageGrid[0][(gridCenterY - rh) * plainWidth[0] + gridCenterX + rw - 1].x, (float)plainImageGrid[0][(gridCenterY - rh) * plainWidth[0] + gridCenterX + rw - 1].y},
//                    {(float)plainImageGrid[0][(gridCenterY + rh - 1) * plainWidth[0] + gridCenterX - rw].x, (float)plainImageGrid[0][(gridCenterY + rh - 1) * plainWidth[0] + gridCenterX - rw].y},
//                    {(float)plainImageGrid[0][(gridCenterY + rh - 1) * plainWidth[0] + gridCenterX + rw - 1].x, (float)plainImageGrid[0][(gridCenterY + rh - 1) * plainWidth[0] + gridCenterX + rw - 1].y},
//            };
//            const std::vector<cv::Point2f> dest = {
//                    {(float)plainImageGrid[1][(gridCenterY2 - rh) * plainWidth[1] + gridCenterX2 - rw].x, (float)plainImageGrid[1][(gridCenterY2 - rh) * plainWidth[0] + gridCenterX2 - rw].y},
//                    {(float)plainImageGrid[1][(gridCenterY2 - rh) * plainWidth[1] + gridCenterX2 + rw - 1].x, (float)plainImageGrid[1][(gridCenterY2 - rh) * plainWidth[0] + gridCenterX2 + rw - 1].y},
//                    {(float)plainImageGrid[1][(gridCenterY2 + rh - 1) * plainWidth[1] + gridCenterX2 - rw].x, (float)plainImageGrid[1][(gridCenterY2 + rh - 1) * plainWidth[0] + gridCenterX2 - rw].y},
//                    {(float)plainImageGrid[1][(gridCenterY2 + rh - 1) * plainWidth[1] + gridCenterX2 + rw - 1].x, (float)plainImageGrid[1][(gridCenterY2 + rh - 1) * plainWidth[0] + gridCenterX2 + rw - 1].y},
//            };
//            transform = cv::getPerspectiveTransform(src, dest);
//
//            CV_Assert(transform.type() == CV_64FC1);
//
//            bestMap1[1].copyTo(alignedMap);
        }

    double Calibrator::getFx() const {
        return fx;
    }

    double Calibrator::getFy() const {
        return fy;
    }
} // ecv