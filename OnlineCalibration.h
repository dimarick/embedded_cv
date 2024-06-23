#ifndef EMBEDDED_CV_ONLINECALIBRATION_H
#define EMBEDDED_CV_ONLINECALIBRATION_H

#include <StereoCalib.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

#define ONLINE_CALIBRATION_MAX_IMAGES 10

#define ONLINE_CALIBRATION_BASE_DIR "./"
#define ONLINE_CALIBRATION_FILE_INTRINSICS 	ONLINE_CALIBRATION_BASE_DIR "data/intrinsics.yml"
#define ONLINE_CALIBRATION_FILE_EXTRINSICS 	ONLINE_CALIBRATION_BASE_DIR "data/extrinsics.yml"
#define ONLINE_CALIBRATION_FILE_CH_BOARD_L 	ONLINE_CALIBRATION_BASE_DIR "data/chbL%0d.png"
#define ONLINE_CALIBRATION_FILE_CH_BOARD_R 	ONLINE_CALIBRATION_BASE_DIR "data/chbR%0d.png"

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
