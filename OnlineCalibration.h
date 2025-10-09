#ifndef EMBEDDED_CV_ONLINECALIBRATION_H
#define EMBEDDED_CV_ONLINECALIBRATION_H

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stereo.hpp"
#include "opencv2/3d.hpp"
#include <iostream>

using namespace cv;

#define ONLINE_CALIBRATION_MAX_IMAGES 10

#define ONLINE_CALIBRATION_BASE_DIR "./"
#define ONLINE_CALIBRATION_FILE_INTRINSICS 	ONLINE_CALIBRATION_BASE_DIR "data/intrinsics.yml"
#define ONLINE_CALIBRATION_FILE_EXTRINSICS 	ONLINE_CALIBRATION_BASE_DIR "data/extrinsics.yml"
#define ONLINE_CALIBRATION_FILE_CH_BOARD_L 	ONLINE_CALIBRATION_BASE_DIR "data/chbL%0d.png"
#define ONLINE_CALIBRATION_FILE_CH_BOARD_R 	ONLINE_CALIBRATION_BASE_DIR "data/chbR%0d.png"

struct StereoCameraProperties{
    Mat cameraMatrix[2];
    Mat distCoeffs[2];
    Mat R, T, E, F; //rotation matrix, translation vector, essential matrix E=[T*R], fundamental matrix
    Mat R1, R2, P1, P2, Q;
    Size imgSize;
    Rect roi[2]; //region of interest
};

class OnlineCalibration {
public:
    bool isCalibrated = false;
    Mat rmapL[2];
    Mat rmapR[2];
    Size imageSize;
    StereoCameraProperties props;
    std::vector<UMat> imagesL;
    std::vector<UMat> imagesR;
    std::vector<std::vector<Point2f>> imagePointsL;
    std::vector<std::vector<Point2f>> imagePointsR;
    std::vector<std::vector<Point3f>> objectPoints;

    int cornerSubPixSize = 7;
    double calibrateEPS = 1e-2;
    int calibrateCount = 10;

    bool isVerticalStereo;
    bool autoDetectBoard(const UMat &left, const UMat &right);
    void drawChessboardCorners(UMat &left, UMat &right);
    double updateCalibrationResult();
    bool loadCalibrationResult();
    bool storeCalibrationResult();
    void rectify();

    void rejectLastBoard();
    void storeCalibrationImages();
    void loadCalibrationImages();

    void setHighPrecision() {
        cornerSubPixSize = 11;
        calibrateEPS = 1e-7;
        calibrateCount = 10000;
    }

private:
    int imageCount = 0;
    int nextImage = 0;
    int boardWidth = 27;
    int boardHeight = 9;
};

#endif //EMBEDDED_CV_ONLINECALIBRATION_H
