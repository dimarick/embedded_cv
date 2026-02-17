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

        auto random = std::rand() % 8;

        int baseFlags =
                cv::CALIB_USE_LU|
                cv::CALIB_RATIONAL_MODEL|
                cv::CALIB_TILTED_MODEL|
                cv::CALIB_THIN_PRISM_MODEL|
                cv::CALIB_FIX_ASPECT_RATIO|
                cv::CALIB_USE_INTRINSIC_GUESS;

        double result = 0;

        std::vector<double> distCoeff;
        for (int i = 0; i < distCoeff.size(); ++i) {
            distCoeff[i] = data.distCoeff[i];
        }

        try {
            result = cv::calibrateCamera(objectPointsSlice, imagePointsSlice, frameSize, data.cameraMatrix,
                                data.distCoeff, data.rvecs, data.tvecs, baseFlags | flags, term);
        } catch (const std::exception &e) {
            std::cerr << "cv::calibrateCamera failed" << std::endl;

            return 1. / 0.;
        }

//        std::cout << std::format("result = {}, camera.fx = {}, camera.fy = {}, camera.cx = {}, camera.cy = {}, distCoeff = {}\n",
//                                 result, cameraData[0], cameraData[4], cameraData[2], cameraData[5], data.distCoeff);

        double ema = 2. / (21. + 1.);
        fx = ema * cameraData[0] + (1 - ema) * fx;
        fy = ema * cameraData[4] + (1 - ema) * fy;
        cx = ema * cameraData[2] + (1 - ema) * cx;
        cy = ema * cameraData[5] + (1 - ema) * cy;

        for (int i = 0; i < distCoeff.size(); ++i) {
            data.distCoeff[i] = ema * data.distCoeff[i] + (1 - ema) * distCoeff[i];
        }

        return result;
    }

    std::tuple<cv::Mat, cv::Mat, double> Calibrator::getStereoUndistortMap(cv::Size frameSize, const CalibrationData &base, const CalibrationData &current, double alpha) {
        cv::Rect roi1, roi2;
        cv::Mat  baseMap, map, tmp, R1, P1, R2, P2, Q;

//        cv::Mat newBaseCamMat = cv::getOptimalNewCameraMatrix(base.cameraMatrix, base.distCoeff, frameSize, alpha, frameSize, &roi1);
//        cv::Mat newCamMat = cv::getOptimalNewCameraMatrix(current.cameraMatrix, current.distCoeff, frameSize, alpha, frameSize, &roi2);
//        cv::Mat avgMat = (newBaseCamMat + newCamMat) / 2;

        cv::stereoRectify(
                base.cameraMatrix, base.distCoeff,
                current.cameraMatrix, current.distCoeff,
                frameSize, current.R, current.T,
                R1, R2, P1, P2, Q,
                cv::CALIB_ZERO_DISPARITY, alpha, frameSize, &roi1, &roi2
        );
        cv::Mat rvec;
        cv::Rodrigues(current.R, rvec);

//        std::cout << "rvec (rad): " << rvec.t() << std::endl;
//        std::cout << "T (mm): " << current.T.t() << std::endl;
//
//        std::cout << "cam1:\n" << base.cameraMatrix << std::endl;
//        std::cout << "cam2:\n" << current.cameraMatrix << std::endl;
//
//        std::cout << "P1:\n" << P1 << std::endl;
//        std::cout << "P2:\n" << P2 << std::endl;
//        std::cout << "R1:\n" << R1 << std::endl;
//        std::cout << "R2:\n" << R2 << std::endl;

//        printStereoCalibrationStats(base.cameraMatrix, current.cameraMatrix, current.R, current.T);

        cv::initUndistortRectifyMap(base.cameraMatrix, base.distCoeff, R2, P2, frameSize, CV_32FC2, baseMap, tmp);
//        cv::stereoRectify(
//                base.cameraMatrix, base.distCoeff,
//                current.cameraMatrix, current.distCoeff,
//                frameSize, R1.t() * R2, current.T,
//                R1, R2, P1, P2, Q,
//                cv::CALIB_ZERO_DISPARITY, 0.0, frameSize, &roi1, &roi2
//        );

        cv::initUndistortRectifyMap(current.cameraMatrix, current.distCoeff, R1, P1, frameSize, CV_32FC2, map, tmp);

        return {baseMap, map, roi1.width * roi1.height};
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

    double Calibrator::calibrateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3d>> &collectedObjectPoints,
                                             const std::vector<std::vector<cv::Point3d>> &collectedImagePoints,
                                             const std::vector<cv::Point3d> &newObjectPoints,
                                             const std::vector<cv::Point3d> &newImagePoints,
                                             CalibrationData &data,
                                             int flags,
                                             cv::TermCriteria term) {

        cv::Mat mat;
        std::vector<std::vector<cv::Point3d>> objectGrids(collectedObjectPoints.size());
        std::vector<std::vector<cv::Point3d>> imageGrids(collectedImagePoints.size());
        std::copy(collectedImagePoints.begin(), collectedImagePoints.end(), imageGrids.begin());
        std::copy(collectedObjectPoints.begin(), collectedObjectPoints.end(), objectGrids.begin());
        objectGrids.emplace_back(newObjectPoints);
        imageGrids.emplace_back(newImagePoints);

        return calibrateSingleCamera(frameSize, objectGrids, imageGrids, data, flags, term);
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
} // ecv