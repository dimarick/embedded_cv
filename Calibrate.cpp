#include <highgui.hpp>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "Calibrate.h"

void Calibrate::capture(cv::VideoCapture left, cv::VideoCapture right) {
    char imageLoc[200];

    cv::Mat imageLeft, imageRight, imageLeftRot, imageRightRot, imageLeftSm, imageRightSm;

    int key = '0';

    for( int i = 0; i < 10;) {
        left.read(imageLeft);
        right.read(imageRight);

        imageLeft.copyTo(imageLeftRot);
        imageRight.copyTo(imageRightRot);
//
//        cv::rotate(imageLeft, imageLeftRot, cv::ROTATE_180);
//        cv::rotate(imageRight, imageRightRot, cv::ROTATE_180);

        if(imageLeft.empty() || imageRight.empty())
        {
            throw std::runtime_error("Cannot capture data from video source");
        }

        if(key == 'r')
        {
            sprintf(imageLoc, FILE_TEMPLATE_LEFT, i);
            imwrite(imageLoc, imageRight);

            sprintf(imageLoc, FILE_TEMPLATE_RIGHT, i);
            imwrite(imageLoc, imageLeft);

            i++;

            std::cerr << "Captured " << imageLoc << std::endl;
        }

        cv::resize(imageLeftRot, imageLeftSm, cv::Size(320, 240));
        cv::resize(imageRightRot, imageRightSm, cv::Size(320, 240));

        imshow("Left", imageLeftSm);
        imshow("Right", imageRightSm);
        key = cv::waitKey(1);
        if(key == 'q') {
            break;
        }
    }
}

void Calibrate::calibrate(int boardWidth, int boardHeight, StereoCameraProperties& props) {
    cv::Size boardSize;
    boardSize.width = boardWidth;
    boardSize.height = boardHeight;
    bool useCalibrated = true;
    bool showRectified = true;

    //imagelistfn = "data/stereo_calib.xml";

    std::vector<std::string> imagelist;
    bool ok = readStringList("stereo_calib.xml", imagelist);
    if(!ok || imagelist.empty())
    {
        std::cout << "can not open " << "stereo_calib.xml" << " or the string list is empty" << std::endl;
    }

    StereoCalib(imagelist, boardSize, props, useCalibrated, showRectified);
}

void Calibrate::readCalibrationData(StereoCameraProperties &props) {
    //#####################################################################
    //# Camera Setup - load existing intrinsic & extrinsic parameters
    //#####################################################################
    std::string intrinsicFilename = FILE_INTRINSICS;
    std::string extrinsicFilename = FILE_EXTRINSICS;

    // Read in intrinsic parameters
    printf("Loading intrinsic parameters.\n");
    FileStorage fs(intrinsicFilename, FileStorage::READ);
    if(!fs.isOpened())
    {
        throw std::runtime_error("Failed to open file");
    }
    fs["M1"] >> props.cameraMatrix[0];
    fs["D1"] >> props.distCoeffs[0];
    fs["M2"] >> props.cameraMatrix[1];
    fs["D2"] >> props.distCoeffs[1];

    // Read in extrinsic parameters
    printf("Loading extrinsic parameters.\n");
    fs.open(extrinsicFilename, FileStorage::READ);
    if(!fs.isOpened())
    {
        throw std::runtime_error("Failed to open file");
    }

    fs["R"] >> props.R;

    fs["T"] >> props.T;
}
