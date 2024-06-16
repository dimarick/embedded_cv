//
// Created by dima on 15.06.24.
//

#ifndef EMBEDDED_CV_CALIBRATE_H
#define EMBEDDED_CV_CALIBRATE_H

#include <videoio.hpp>
#include <StereoCalib.h>

#define BASE_DIR "./"
// Recalibrate
#define FILE_INTRINSICS 	BASE_DIR "data/intrinsics.yml"
#define FILE_EXTRINSICS 	BASE_DIR "data/extrinsics.yml"
#define FILE_CALIB_XML  	BASE_DIR "data/stereo_calib.xml"
// Recapture
#define FILE_TEMPLATE_LEFT	BASE_DIR "data/chessboard%dL.png"
#define FILE_TEMPLATE_RIGHT	BASE_DIR "data/chessboard%dR.png"

class Calibrate {
public:
    static void capture(cv::VideoCapture left, cv::VideoCapture right);
    static void calibrate(int boardWidth, int boardHeight, StereoCameraProperties& props);
    static void readCalibrationData(StereoCameraProperties& props);
};


#endif //EMBEDDED_CV_CALIBRATE_H
