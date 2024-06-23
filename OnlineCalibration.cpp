#include "OnlineCalibration.h"

bool OnlineCalibration::autoDetectBoard(const UMat &left, const UMat &right) {
    int flags = CALIB_CB_FAST_CHECK | CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
    const auto &termCriteria = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 50, 0.005);

    UMat leftThumb, leftGreen, rightGreen;
    cv::extractChannel(left, leftGreen, 1);
    cv::extractChannel(right, rightGreen, 1);

    imagePointsL.resize(std::min(imageCount + 1, ONLINE_CALIBRATION_MAX_IMAGES));
    imagePointsR.resize(std::min(imageCount + 1, ONLINE_CALIBRATION_MAX_IMAGES));

    imagesL.resize(imagePointsL.size());
    imagesR.resize(imagePointsR.size());

    if (boardHeight > 0) {
        const auto &patternSize = Size(boardWidth, boardHeight);
        if (!cv::findChessboardCorners(left, patternSize, imagePointsL[nextImage], flags)) {
            return false;
        }

        if (!cv::findChessboardCorners(right, patternSize, imagePointsR[nextImage], flags)) {
            return false;
        }

        if (cornerSubPixSize > 0) {
            cornerSubPix(leftGreen, imagePointsL[nextImage], Size(cornerSubPixSize, cornerSubPixSize), Size(-1, -1),
                         termCriteria);
            cornerSubPix(rightGreen, imagePointsR[nextImage], Size(cornerSubPixSize, cornerSubPixSize), Size(-1, -1),
                         termCriteria);
        }

        left.copyTo(imagesL[nextImage]);
        right.copyTo(imagesR[nextImage]);

        nextImage++;
        imageCount++;

        nextImage = nextImage % ONLINE_CALIBRATION_MAX_IMAGES;

        imageSize = left.size();

        return true;
    }

    std::vector<Point2f> corners[16*16];
    cv::resize(left, leftThumb, Size(0, 0), 1. / 3, 1. / 3);

    imshow("thumb", leftThumb);

    for (int i = 3; i < 16; ++i) {
        for (int j = 3; j < 16; ++j) {
            if ((float)j / (float)i > 2.5 || (float)i / (float)j > 2.5) {
                continue;
            }

            const auto &patternSize = Size(i, j);
            if (cv::findChessboardCorners(leftThumb, patternSize, *corners, flags) &&
                cv::findChessboardCorners(left, patternSize, imagePointsL[nextImage], flags) &&
                cv::findChessboardCorners(right, patternSize, imagePointsR[nextImage], flags)) {

                std::cerr << "Detected chessboard " << i << "x" << j << std::endl;

                boardWidth = i;
                boardHeight = j;

                if (cornerSubPixSize > 0) {
                    cornerSubPix(leftGreen, imagePointsL[nextImage], Size(cornerSubPixSize, cornerSubPixSize),
                                 Size(-1, -1), termCriteria);
                    cornerSubPix(rightGreen, imagePointsR[nextImage], Size(cornerSubPixSize, cornerSubPixSize),
                                 Size(-1, -1), termCriteria);
                }

                left.copyTo(imagesL[nextImage]);
                right.copyTo(imagesR[nextImage]);

                nextImage++;
                imageCount++;

                nextImage = nextImage % ONLINE_CALIBRATION_MAX_IMAGES;

                return true;
            }
        }
    }

    return false;
}

void OnlineCalibration::drawChessboardCorners(UMat &left, UMat &right) {
    int i = nextImage;

    if (imageCount == 0) {
        return;
    }

    if (i <= 0) {
        i = ONLINE_CALIBRATION_MAX_IMAGES;
    }

    const auto &patternSize = Size(boardWidth, boardHeight);
    cv::drawChessboardCorners(left, patternSize, imagePointsL[i - 1], true);
    cv::drawChessboardCorners(right, patternSize, imagePointsR[i - 1], true);
}

double OnlineCalibration::updateCalibrationResult() {
    int nImages = std::min(ONLINE_CALIBRATION_MAX_IMAGES, imageCount);

    if (nImages == 0) {
        return INFINITY;
    }

    float squareSize = 1.;
    objectPoints.resize(nImages);
    for (int i = 0; i < nImages; i++) {
        objectPoints[i].resize(boardHeight * boardWidth);
        for( int j = 0; j < boardHeight; j++ ) {
            for( int k = 0; k < boardWidth; k++ ) {
                objectPoints[i][j * boardWidth + k] = cv::Point3f((float)k * squareSize, (float)j * squareSize, 0);
            }
        }
    }

    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;

    double rms = INFINITY;

    try {
        rms = cv::stereoCalibrate(objectPoints, imagePointsL, imagePointsR,
                                                cameraMatrix[0], distCoeffs[0],
                                                cameraMatrix[1], distCoeffs[1],
                                                imageSize, R, T, E, F,
                                                CALIB_SAME_FOCAL_LENGTH +
                                                CALIB_FIX_SKEW +
                                                CALIB_RECOMPUTE_EXTRINSIC +
                                                CALIB_USE_LU +
                                                CALIB_USE_QR ,
                                                TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, calibrateCount,
                                                             calibrateEPS));

    } catch (const std::exception &e) {
        std::cerr << "Exception on calibrate" << std::endl;
        return rms;
    }

    std::cerr << "Calibrated with RMS " << rms << std::endl;

    if (rms == NAN || rms > 10) {
        std::cerr << "Failed to calibrate" << std::endl;

        return rms;
    }

// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    std::vector<Vec3f> lines[2];
//    for( int i = 0; i < nImages; i++ )
//    {
//        int npt = (int)imagePointsL[i].size();
//        Mat imgptL = Mat(imagePointsL[i]);
//        Mat imgptR = Mat(imagePointsR[i]);
//        undistortPoints(imgptL, imgptL, cameraMatrix[0], distCoeffs[0], Mat(), cameraMatrix[0]);
//        undistortPoints(imgptR, imgptR, cameraMatrix[1], distCoeffs[1], Mat(), cameraMatrix[1]);
//        computeCorrespondEpilines(imgptL, 1, F, lines[0]);
//        computeCorrespondEpilines(imgptR, 2, F, lines[1]);
//
//        for( int j = 0; j < npt; j++ )
//        {
//            double errij = fabs(imagePointsL[i][j].x*lines[1][j][0] +
//                                imagePointsL[i][j].y*lines[1][j][1] + lines[1][j][2]) +
//                           fabs(imagePointsR[i][j].x*lines[0][j][0] +
//                                imagePointsR[i][j].y*lines[0][j][1] + lines[0][j][2]);
//            err += errij;
//        }
//        npoints += npt;
//    }
//    std::cout << "average reprojection err = " <<  err/npoints << std::endl;

    props.cameraMatrix[0] = cameraMatrix[0];
    props.cameraMatrix[1] = cameraMatrix[1];
    props.distCoeffs[0] = distCoeffs[0];
    props.distCoeffs[1] = distCoeffs[1];
    props.R = R;
    props.T = T;
    props.E = E;
    props.F = F;

    return rms;
}

bool OnlineCalibration::loadCalibrationResult() {
    //#####################################################################
    //# Camera Setup - load existing intrinsic & extrinsic parameters
    //#####################################################################
    std::string intrinsicFilename = ONLINE_CALIBRATION_FILE_INTRINSICS;
    std::string extrinsicFilename = ONLINE_CALIBRATION_FILE_EXTRINSICS;

    // Read in intrinsic parameters
    printf("Loading intrinsic parameters.\n");
    FileStorage fs(intrinsicFilename, FileStorage::READ);
    if(!fs.isOpened()) {
        std::cerr << "Failed to open file" << std::endl;
        return false;
    }
    fs["M1"] >> props.cameraMatrix[0];
    fs["D1"] >> props.distCoeffs[0];
    fs["M2"] >> props.cameraMatrix[1];
    fs["D2"] >> props.distCoeffs[1];

    // Read in extrinsic parameters
    printf("Loading extrinsic parameters.\n");
    fs.open(extrinsicFilename, FileStorage::READ);
    if(!fs.isOpened()) {
        std::cerr << "Failed to open file" << std::endl;
        return false;
    }

    fs["R"] >> props.R;
    fs["T"] >> props.T;
    fs["W"] >> imageSize.width;
    fs["H"] >> imageSize.height;

    rectify();

    return true;
}

void OnlineCalibration::storeCalibrationImages() {
    char buffer[1000];
    for (int i = 0; i < imagesL.size(); ++i) {
        sprintf(buffer, ONLINE_CALIBRATION_FILE_CH_BOARD_L, i);
        imwrite(buffer, imagesL[i]);
        sprintf(buffer, ONLINE_CALIBRATION_FILE_CH_BOARD_R, i);
        imwrite(buffer, imagesR[i]);
    }
}

void OnlineCalibration::loadCalibrationImages() {
    char buffer[1000];
    UMat l, r;
    for (int i = 0; i < ONLINE_CALIBRATION_MAX_IMAGES; ++i) {
        sprintf(buffer, ONLINE_CALIBRATION_FILE_CH_BOARD_L, i);
        Mat mat = imread(buffer);
        mat.copyTo(l);
        sprintf(buffer, ONLINE_CALIBRATION_FILE_CH_BOARD_R, i);
        mat = imread(buffer);
        mat.copyTo(r);

        if (l.empty() || r.empty()) {
            break;
        }

        if (!autoDetectBoard(l, r)) {
            continue;
        }
    }
}

bool OnlineCalibration::storeCalibrationResult() {

    // save intrinsic parameters
    FileStorage fs(ONLINE_CALIBRATION_FILE_INTRINSICS, FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << props.cameraMatrix[0] << "D1" << props.distCoeffs[0] <<
           "M2" << props.cameraMatrix[1] << "D2" << props.distCoeffs[1];
        fs.release();
    }
    else{
        std::cout << "Error: can not save the intrinsic parameters " << std::endl;
    }

    fs.open(ONLINE_CALIBRATION_FILE_EXTRINSICS, FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << props.R << "T" << props.T << "W" << imageSize.width << "H" << imageSize.height;
        fs.release();
    }
    else{
        std::cout << "Error: can not save the extrinsic parameters" << std::endl;
    }

    return true;
}

void OnlineCalibration::rectify() {
    if (props.R.empty()) {
        return;
    }

    cv::stereoRectify(props.cameraMatrix[0], props.distCoeffs[0],
                  props.cameraMatrix[1], props.distCoeffs[1],
                  imageSize, props.R, props.T, props.R1, props.R2, props.P1, props.P2, props.Q,
                  0, 0.5, props.imgSize, &props.roi[0], &props.roi[1]);

    // OpenCV can handle left-right
    // or up-down camera arrangements
    isVerticalStereo = fabs(props.P2.at<double>(1, 3)) > fabs(props.P2.at<double>(0, 3));

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(props.cameraMatrix[0], props.distCoeffs[0], props.R1, props.P1, imageSize, CV_16SC2, rmapL[0], rmapL[1]);
    initUndistortRectifyMap(props.cameraMatrix[1], props.distCoeffs[1], props.R2, props.P2, imageSize, CV_16SC2, rmapR[0], rmapR[1]);

    isCalibrated = true;
}

void OnlineCalibration::rejectLastBoard() {
    nextImage--;
    imageCount--;

    if (imageCount < 0) {
        imageCount = 0;
    }

    if (nextImage < 0) {
        nextImage = ONLINE_CALIBRATION_MAX_IMAGES - 1;
    }
}
