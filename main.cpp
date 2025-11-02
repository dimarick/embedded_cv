#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif
#include "opencv2/videoio.hpp"
#include "ImageProcessor.h"
#include "CvCommandHandler.h"
#include "OnlineCalibration.h"
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <chrono>
#include <cpptrace/cpptrace.hpp>
#include <CommandServer.h>
#include <BroadcastingServer.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <SocketFactory.h>
#include <csignal>
#include <core/opencl/ocl_defs.hpp>
#include <fstream>

#include "QuickStereoMatch.h"
#include "CalibrateMapper.h"
#include "Calibrator.h"

using namespace mini_server;

void drawText(UMat &image, double rms);

static std::atomic running = true;

static BroadcastingServer broadcastingServer;
static CommandServer commandServer;
int findNearestPoint(const Point2f &point, const std::vector<Point3d> &points, double searchRadius);


void matwrite(const std::string& filename, const Mat& mat)
{
    std::ofstream fs(filename, std::fstream::binary);

    // Header
    int type = mat.type();
    int channels = mat.channels();// rows
    fs.write((char*)&mat.rows, sizeof(int));    // rows
    fs.write((char*)&mat.cols, sizeof(int));    // cols
    fs.write((char*)&type, sizeof(int));        // type
    fs.write((char*)&channels, sizeof(int));    // channels

    // Data
    if (mat.isContinuous()) {
        fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
    } else {
        int rowsz = CV_ELEM_SIZE(type) * mat.cols;
        for (int r = 0; r < mat.rows; ++r)
        {
            fs.write(mat.ptr<char>(r), rowsz);
        }
    }
}

void matread(const std::string& filename, Mat &mat)
{
    std::ifstream fs(filename, std::fstream::binary);

    if (!fs.good()) {
        mat = Mat();
        return;
    }

    // Header
    int rows, cols, type, channels;
    fs.read((char*)&rows, sizeof(int));         // rows
    fs.read((char*)&cols, sizeof(int));         // cols
    fs.read((char*)&type, sizeof(int));         // type
    fs.read((char*)&channels, sizeof(int));     // channels

    // Data
    mat = Mat(rows, cols, type);
    fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);
}

void invMap(const cv::Mat &src, cv::Mat &dest) {
    if (dest.empty()) {
        dest = Mat(src.size(), src.type());
    }

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            const auto &p = src.at<cv::Point2f>(y, x);
            dest.at<cv::Point2f>((int)std::round(p.y), (int)std::round(p.x)) = Point2f((float)x, (float)y);
        }
    }
}

int main(int argc, const char **argv) {
    cpptrace::register_terminate_handler();

    cv::Mat captureFrameLeft, captureFrameRight, readingLeft, readingRight, readingImLeft, readingImRight;
    cv::UMat output, outputLeft, inputLeft, imageLeft, outputRight, inputRight, imageRight;
    cv::Mat resultLeft, resultRight;
    cv::VideoCapture captureLeft, captureRight;

    cpptrace::register_terminate_handler();

    std::cerr << (cv::ocl::haveOpenCL() ? "with OpenCl" : "cpu only") << std::endl;

    std::cerr << "Built with OpenCV " << CV_VERSION << ", cv::ocl::haveSVM(): " << cv::ocl::haveSVM() << std::endl;

    std::vector<int> params = {
        cv::VideoCaptureProperties::CAP_PROP_CONVERT_RGB, true,
        cv::VideoCaptureProperties::CAP_PROP_FPS, 30,
        cv::VideoCaptureProperties::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, 1920,
        cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, 1080,
        cv::VideoCaptureProperties::CAP_PROP_BUFFERSIZE, 10,
    };

    if (argc < 3) {
        std::cout << "too less command arguments";

        return -1;
    }

    if (strcmp(argv[1], argv[2]) == 0) {
        captureLeft.open(argv[1], cv::CAP_V4L2, params);
        captureRight = captureLeft;
    } else {
        captureLeft.open(argv[1], cv::CAP_V4L2, params);
        captureRight.open(argv[2], cv::CAP_V4L2, params);
    }
//
//    StereoCameraProperties props;
//    if (strcmp(argv[3], "--recapture-chessboard") == 0) {
//        Calibrate::capture(captureLeft, captureRight);
//        Calibrate::calibrate(12, 8, props);
//        return 0;
//    } else if (strcmp(argv[3], "--recalibrate") == 0) {
//        Calibrate::calibrate(12, 8, props);
//        return 0;
//    } else {
//        Calibrate::readCalibrationData(props);
//        captureLeft.read(captureFrameLeft);
//        props.imgSize = captureFrameLeft.size();
//    }
//    //stereoRectify performed inside calibration function atm but not necessary in finding extrinsics that can be loaded
//    stereoRectify(props.cameraMatrix[0], props.distCoeffs[0], props.cameraMatrix[1], props.distCoeffs[1],
//                  props.imgSize, props.R, props.T, props.R1, props.R2, props.P1, props.P2, props.Q,
//                  CALIB_ZERO_DISPARITY, 1, props.imgSize, &props.roi[0], &props.roi[1]);
//
//    cv::Mat mapl[2], mapr[2];
//    ///Computes the undistortion and rectification transformation map.
//    ///http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#initundistortrectifymap
//    //Left
//    initUndistortRectifyMap(props.cameraMatrix[0], props.distCoeffs[0], props.R1, props.P1, props.imgSize, CV_16SC2, mapl[0], mapl[1]);
//    //Right
//    initUndistortRectifyMap(props.cameraMatrix[1], props.distCoeffs[1], props.R2, props.P2, props.imgSize, CV_16SC2, mapr[0], mapr[1]);

    broadcastingServer.setSocket(SocketFactory::createListeningSocket("/tmp/cv_tm", 10));
    commandServer.setSocket(SocketFactory::createListeningSocket("/tmp/cv_ctl", 1));
    auto handler = CvCommandHandler(commandServer);
    commandServer.setHandler(handler);

    signal(SIGINT, [](int signal) {
        running = false;
        broadcastingServer.stop();
        commandServer.stop();
    });

    std::thread commandServerThread = std::thread([]() {
        commandServer.run();
    });

    std::thread broadcastingServerThread = std::thread([]() {
        broadcastingServer.run();
    });

    auto capFps = captureLeft.get(cv::CAP_PROP_FPS);
    auto capWidth = (int) captureLeft.get(cv::CAP_PROP_FRAME_WIDTH);
    auto capHeight = (int) captureLeft.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto capPixFormat = (int) captureLeft.get(cv::CAP_PROP_PVAPI_PIXELFORMAT);
    auto capBuffer = (int) captureLeft.get(cv::CAP_PROP_BUFFERSIZE);

    const char *humanPixelFormat;

    switch (capPixFormat) {
        case cv::CAP_PVAPI_PIXELFORMAT_MONO8:
            humanPixelFormat = "Mono8";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_MONO16:
            humanPixelFormat = "Mono16";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_BAYER8:
            humanPixelFormat = "Bayer8";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_BAYER16:
            humanPixelFormat = "Bayer16";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_RGB24:
            humanPixelFormat = "Rgb24";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_BGR24:
            humanPixelFormat = "Bgr24";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_RGBA32:
            humanPixelFormat = "Rgba32";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_BGRA32:
            humanPixelFormat = "Bgra32";
            break;
        default:
            humanPixelFormat = "n/a";
    }

    std::cerr << "Capture opened, " << capWidth << "x" << capHeight << "p" << capFps << "p, " << humanPixelFormat
              << ", " << capPixFormat << ", buffer " << capBuffer << std::endl;

    auto prev = std::chrono::high_resolution_clock::now();

    if (!captureLeft.isOpened()) {
        std::cerr << "Left Capture is not opened" << std::endl;
        return -1;
    }

    if (!captureRight.isOpened()) {
        std::cerr << "Right Capture is not opened" << std::endl;
        return -1;
    }

    captureLeft.read(captureFrameLeft);
    captureRight.read(captureFrameRight);

    captureLeft.release();
    captureRight.release();

    cv::Size outputSize(capWidth / 2, capHeight / 2);
    cv::Size processingSize(capWidth, capHeight);

    const char command[] = "ffmpeg -f rawvideo -pixel_format bgr24 -s %dx%d -re  -i - %s %s";
    const char inputCommand[] = "ffmpeg -loglevel fatal -f v4l2 -input_format mjpeg -s %dx%d -re  -i %s -f rawvideo -filter:v 'format=bgr24' -fflags nobuffer -avioflags direct -fflags discardcorrupt -g 15 -threads 7 -";
    size_t bufferSize = sizeof(command) + strlen(argv[3]) + strlen(argv[4]) + 50;
    char *formattedCommand = (char *) malloc(bufferSize);
    snprintf(formattedCommand, bufferSize, command, outputSize.width, outputSize.height, argv[3], argv[4]);

    std::cerr << formattedCommand << std::endl;
    std::cerr.flush();

    auto pipe = popen(formattedCommand, "w");

    free(formattedCommand);

    size_t inputBufferSize = sizeof(inputCommand) + strlen(argv[1]) + strlen(argv[2]) + 50;
    char *formattedInputCommand = (char *) malloc(inputBufferSize);
    snprintf(formattedInputCommand, inputBufferSize, inputCommand, capWidth, capHeight, argv[1]);

    std::cerr << formattedInputCommand << std::endl;
    std::cerr.flush();

    auto leftPipe = popen(formattedInputCommand, "r");

    if (leftPipe == nullptr) {
        std::cerr << "Failed to start " << formattedInputCommand << " errno " << strerror(errno) << std::endl;

        return -1;
    }

    snprintf(formattedInputCommand, inputBufferSize, inputCommand, capWidth, capHeight, argv[2]);

    std::cerr << formattedInputCommand << std::endl;
    std::cerr.flush();

    auto rightPipe = popen(formattedInputCommand, "r");

    if (rightPipe == nullptr) {
        std::cerr << "Failed to start " << formattedInputCommand << " errno " << strerror(errno) << std::endl;

        return -1;
    }

    free(formattedInputCommand);

    if (pipe == nullptr) {
        std::cerr << "Failed to start " << formattedCommand << " errno " << strerror(errno) << std::endl;

        return -1;
    }

    ImageProcessor processor(processingSize.width, processingSize.height, broadcastingServer);

    handler.setImageProcessor(&processor);

    double fps = 0.;
    double avgFps = 0.;
    double avgTime = 0.;

    captureFrameLeft.copyTo(readingLeft);
    captureFrameRight.copyTo(readingRight);

    std::thread writer;
    std::atomic writerIsRunning = false;

    cv::ocl::setUseOpenCL(true);

    std::mutex readerLeftLock;
    std::mutex readerRightLock;
    std::atomic readerLeftCount = 0l;
    std::atomic readerRightCount = 0l;
    auto frameCount = 0l;

    auto readerfunction =
            [](cv::Mat *reading, FILE *pipe, cv::Mat *readingIm, std::mutex *lock, std::atomic<long> *count) {
        while (running) {
            fread(reading->data, sizeof(char), reading->dataend - reading->datastart, pipe);
            lock->lock();
            reading->copyTo(*readingIm);
            (*count)++;
            lock->unlock();
        }
    };
    auto readerLeft = std::thread(readerfunction, &readingLeft, leftPipe, &readingImLeft, &readerLeftLock, &readerLeftCount);
    auto readerRight = std::thread(readerfunction, &readingRight, rightPipe, &readingImRight, &readerRightLock, &readerRightCount);

    if (!cv::ocl::isOpenCLActivated()) {
//        throw std::runtime_error("OpenCL is not available");
    }

    std::atomic calibrating = false;
    std::atomic<double> calibrationRMS = INFINITY;
    std::thread calibration;

    UMat calibLeft, calibRight;

    UMat lFrame, rFrame;
    UMat lDispMap, rDispMap, leftDispMap, rightDispMap;

    QuickStereoMatch sm0, sm1, sm;
//
//    Mat ltest = (Mat_<unsigned char>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, })).reshape(1);
//    Mat rtest = (Mat_<unsigned char>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, })).reshape(1);
//    Mat ldtest;
//    Mat rdtest;
//
//    sm0.computeDisparityMap(ltest, rtest, ldtest, rdtest, 5, 6, 5);
//
//    std::cout << ldtest.rowRange(5, 6) << std::endl;
//    std::cout << rdtest.rowRange(5, 6) << std::endl << std::endl;
//
//    ltest = (Mat_<unsigned char>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, })).reshape(1);
//    rtest = (Mat_<unsigned char>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, })).reshape(1);
//    ldtest.release();
//    rdtest.release();
//
//    sm1.computeDisparityMap(ltest, rtest, ldtest, rdtest, 5, 6, 5);
//
//    std::cout << ldtest.rowRange(5, 6) << std::endl;
//    std::cout << rdtest.rowRange(5, 6) << std::endl << std::endl;

//    running = false;


    std::vector<cv::UMat> frames(2);
    cv::Mat aligned;
    cv::Mat alignedMap;
    cv::Mat invAlignedMap;
    std::vector<cv::Mat> bestMap1(frames.size()), bestMap2(frames.size());
    std::vector<cv::Mat> invBestMap1(frames.size()), invBestMap2(frames.size());

    matread("map0.bin", bestMap1[0]);
    matread("map1.bin", bestMap1[1]);
    matread("mapa.bin", alignedMap);

    bool needsCalibration1 = bestMap1[0].empty() || bestMap1[1].empty() || alignedMap.empty();

    if (!needsCalibration1) {
        for (int i = 0; i < frames.size(); ++i) {
            invMap(bestMap1[i], invBestMap1[i]);
        }
        invMap(alignedMap, invAlignedMap);
    }

    std::vector<double> avgDistCoeff(14);
    std::vector<double> prevStdRmseUndistorted(frames.size(), 1e6);
    std::vector<size_t> nextFrameId(frames.size());

    std::vector<ecv::CalibrateMapper<double>> calibrateMapper(frames.size());
    std::vector<ecv::CalibrateMapper<double>> calibrateMapper2(frames.size());
    std::vector<ecv::CalibrateMapper<double>> calibrateMapper3(frames.size());
    std::vector<ecv::Calibrator> calibrator(frames.size());

    std::vector objectPoints(frames.size(), std::vector<std::vector<Point3f>>(3));
    std::vector imagePoints(frames.size(), std::vector<std::vector<Point2f>>(3));

    namedWindow("Plain best " + std::to_string(0), WINDOW_AUTOSIZE);
    namedWindow("src " + std::to_string(0), WINDOW_AUTOSIZE);

    auto mousePlain = cv::Point2i();
    auto mouseSrc = cv::Point2i();

    cv::setMouseCallback("Plain best " + std::to_string(0), [](int event, int x, int y, int flags, void* userdata) {
        auto mouse = (cv::Point2i *)userdata;
        mouse->x = x;
        mouse->y = y;
    }, &mousePlain);

    cv::setMouseCallback("src " + std::to_string(0), [](int event, int x, int y, int flags, void* userdata) {
        auto mouse = (cv::Point2i *)userdata;
        mouse->x = x;
        mouse->y = y;
    }, &mouseSrc);

    for (int i = 0; running; i++) {
        long nextFrame = std::max(readerLeftCount, readerRightCount);
        if (readerLeftCount == frameCount || readerRightCount == frameCount || readingImLeft.empty() || readingImRight.empty()) {
            usleep(1);
            continue;
        }

        frameCount = nextFrame;

        auto start = std::chrono::high_resolution_clock::now();

        readerLeftLock.lock();
        readingImLeft.copyTo(imageLeft);
        readerLeftLock.unlock();

        readerRightLock.lock();
        readingImRight.copyTo(imageRight);
        readerRightLock.unlock();

        cv::rotate(imageRight, imageRight, ROTATE_180);

        frames = {imageLeft, imageRight};

        auto now = std::chrono::high_resolution_clock::now();
        auto us = (double) (now - prev).count();
        prev = now;

        fps = 1e9 / us;

        if (imageLeft.empty()) {
            break;
        }

        if (needsCalibration1) {
            std::vector<cv::Mat> map1(frames.size()), map2(frames.size());
            std::vector imageGrid(frames.size(), std::vector<Point3d>(1000));
            std::vector objectGrid(frames.size(), std::vector<Point3d>(imageGrid[0].size()));
            std::vector<size_t> gridWidth(frames.size());
            std::vector<size_t> gridHeight(frames.size());
            std::vector<ecv::CalibrateMapper<double>::BaseSquare> square(frames.size());
            std::vector<cv::Mat> plain(frames.size());

            for (int j = 0; j < frames.size(); ++j) {
                Mat colorFrame;
                const auto &frame = frames[j].getMat(AccessFlag::ACCESS_RW);
                frame.copyTo(colorFrame);


                auto gridRmse = calibrateMapper[j].detectFrameImagePointsGrid(colorFrame, imageGrid[j], &gridWidth[j], &gridHeight[j], colorFrame);

                auto stdGridRmse = 1e6;
                if (calibrateMapper[j].isGridValid(colorFrame.size(), imageGrid[j], gridWidth[j], gridHeight[j])) {
                    stdGridRmse = calibrateMapper[j].generateFrameObjectPointsGrid2(colorFrame.size(), imageGrid[j], objectGrid[j], gridWidth[j],
                                                                                gridHeight[j]);
                }

                calibrateMapper[j].drawGrid(colorFrame, imageGrid[j], gridWidth[j], gridHeight[j], cv::Scalar(255, 0, 0));
                calibrateMapper[j].drawGrid(colorFrame, objectGrid[j], gridWidth[j], gridHeight[j], cv::Scalar(255, 255, 0));
                calibrateMapper[j].drawGridCorrelation(colorFrame, imageGrid[j], objectGrid[j], gridWidth[j], gridHeight[j], cv::Scalar(255, 0, 255));

                std::cerr << "; gridRmse " << gridRmse << "; stdGridRmse " << stdGridRmse << "; patternSize " << calibrateMapper[j].patternSize << "; skew " << calibrateMapper[j].skew << std::endl;

                if (
                        calibrateMapper[j].isGridValid(colorFrame.size(), imageGrid[j], gridWidth[j], gridHeight[j])
                        && calibrateMapper[j].isGridValid(colorFrame.size(), objectGrid[j], gridWidth[j], gridHeight[j])
                        ) {
                    auto frameId = nextFrameId[j] % objectPoints[j].size();
                    nextFrameId[j]++;

                    imagePoints[j][frameId].resize(gridWidth[j] * gridHeight[j]);
                    objectPoints[j][frameId].resize(gridWidth[j] * gridHeight[j]);

                    calibrateMapper[j].convertTo2dPoints(imageGrid[j], imagePoints[j][frameId]);
                    calibrateMapper[j].convertToPlain3dPoints(objectGrid[j], objectPoints[j][frameId]);

                    calibrator[j].calibrate(colorFrame.size(), objectPoints[j], imagePoints[j], map1[j], map2[j]);
                }

                if (!map1[j].empty()) {
                    Mat plainCurrent;
                    cv::remap(frame, plainCurrent, map1[j], map2[j], INTER_LANCZOS4);

                    std::vector<Point3d> plainImageGrid(1000);
                    std::vector<Point3d> plainObjectGrid(plainImageGrid.size());
                    size_t plainGridWidth = 0;
                    size_t plainGridHeight = 0;

                    auto plainGridRmse = calibrateMapper2[j].detectFrameImagePointsGrid(plainCurrent, plainImageGrid, &plainGridWidth, &plainGridHeight, plainCurrent);

                    auto plainStdGridRmse = 1e6;
                    if (calibrateMapper2[j].isGridValid(plainCurrent.size(), plainImageGrid, plainGridWidth, plainGridHeight)) {
                        plainStdGridRmse = calibrateMapper2[j].generateFrameObjectPointsGrid(plainCurrent.size(), plainImageGrid, plainObjectGrid, plainGridWidth,
                                                                                             plainGridHeight);
                    }

                    if (plainStdGridRmse < prevStdRmseUndistorted[j] && plainGridWidth == gridWidth[j] && plainGridHeight == gridHeight[j]) {
                        prevStdRmseUndistorted[j] = plainStdGridRmse;
                        map1[j].copyTo(bestMap1[j]);
                        map2[j].copyTo(bestMap2[j]);
                    } else {
                        nextFrameId[j]++;
                    }
                    std::cout << "; plainGridRmse " << plainGridRmse << "; stdGridRmse " << plainStdGridRmse << "; prevStdRmseUndistorted " << prevStdRmseUndistorted[j] << std::endl;

//                    calibrateMapper2[j].drawGrid(plainCurrent, plainImageGrid, plainGridWidth, plainGridHeight, cv::Scalar(255, 0, 0), 1);
//                    calibrateMapper2[j].drawGrid(plainCurrent, plainObjectGrid, plainGridWidth, plainGridHeight, cv::Scalar(255, 255, 0), 1);
//                calibrateMapper.drawGridCorrelation(plainLeft, plainImageGrid, plainObjectGrid, plainGridWidth, plainGridHeight, cv::Scalar(255, 0, 255));

//                    imshow("Plain " + std::to_string(j), plain);
                }
                imshow("GL " + std::to_string(j), colorFrame);

                if (!bestMap1[j].empty()) {
                    cv::remap(frame, plain[j], bestMap1[j], bestMap2[j], INTER_NEAREST);

                    imshow("Plain best " + std::to_string(j), plain[j]);
                }
            }

            bool isAllCalibrated = true;

            for (int j = 0; j < frames.size(); ++j) {
                isAllCalibrated &= !plain[j].empty();
            }

            if (frames.size() >= 2 && isAllCalibrated) {
                std::vector plainImageGrid(frames.size(), std::vector<Point3d>(1000));
                std::vector<size_t> plainWidth(frames.size(), 0);
                std::vector<size_t> plainHeight(frames.size(), 0);

                for (int j = 0; j < frames.size(); ++j) {
                    const auto &frame = frames[j].getMat(AccessFlag::ACCESS_RW);

                    auto plainGridRmse = calibrateMapper3[j].detectFrameImagePointsGrid(plain[j], plainImageGrid[j], &plainWidth[j], &plainHeight[j]);

                    if (std::isnan(plainGridRmse)) {
                        break;
                    }
                }

                if (plainWidth[0] == plainWidth[1] && plainHeight[0] == plainHeight[1] && plainHeight[0] > 0 && plainHeight[1] > 0) {
                    const auto &baseFrame = plain[0];
                    const auto &otherFrame = plain[1];

                    const auto &baseSize = baseFrame.size();
                    cv::Point2f imageCenter = {(float)(baseSize.width - 1) / 2, (float)(baseSize.height - 1) / 2};
                    auto gridCenter = findNearestPoint(imageCenter, plainImageGrid[0], (double)calibrateMapper3[0].patternSize);
                    auto gridCenter2 = findNearestPoint(imageCenter, plainImageGrid[1], (double)calibrateMapper3[1].patternSize);

                    if (gridCenter < 0 || gridCenter2 < 0) {
                        continue;
                    }

                    cv::Mat transform;

                    if (false) {
                        auto gridCenterX = (size_t) gridCenter % plainWidth[0];
                        auto gridCenterY = (size_t) gridCenter / plainWidth[0];
                        auto gridCenterX2 = (size_t) gridCenter2 % plainWidth[1];
                        const std::vector<cv::Point2f> src = {
                                {(float) plainImageGrid[0][gridCenterY * plainWidth[0] +
                                                           gridCenterX].x,                                     (float) plainImageGrid[0][
                                        gridCenterY * plainWidth[0] + gridCenterX].y},
                                {(float) plainImageGrid[0][(gridCenterY - 1) * plainWidth[0] +
                                                           gridCenterX].x,                                     (float) plainImageGrid[0][
                                        (gridCenterY - 1) * plainWidth[0] + gridCenterX].y},
                                {(float) plainImageGrid[0][gridCenterY * plainWidth[0] + gridCenterX -
                                                           1].x,                                               (float) plainImageGrid[0][
                                        gridCenterY * plainWidth[0] + gridCenterX - 1].y},
                        };
                        const std::vector<cv::Point2f> dest = {
                                {(float) plainImageGrid[1][gridCenterY * plainWidth[0] +
                                                           gridCenterX2].x,                                     (float) plainImageGrid[1][
                                        gridCenterY * plainWidth[0] + gridCenterX2].y},
                                {(float) plainImageGrid[1][(gridCenterY - 1) * plainWidth[0] +
                                                           gridCenterX2].x,                                     (float) plainImageGrid[1][
                                        (gridCenterY - 1) * plainWidth[0] + gridCenterX2].y},
                                {(float) plainImageGrid[1][gridCenterY * plainWidth[0] + gridCenterX2 -
                                                           1].x,                                                (float) plainImageGrid[1][
                                        gridCenterY * plainWidth[0] + gridCenterX2 - 1].y},
                        };
                        transform = cv::getAffineTransform(src, dest);
                    } else {
                        auto gridCenterX = (size_t) gridCenter % plainWidth[0];
                        auto gridCenterY = (size_t) gridCenter / plainWidth[0];
                        auto gridCenterX2 = (size_t) gridCenter2 % plainWidth[1];
                        auto gridCenterY2 = gridCenterY;
                        auto rw = std::min({
                            gridCenterX,
                            gridCenterX2,
                            plainWidth[0] - gridCenterX,
                            plainWidth[1] - gridCenterX2,
                        });
                        auto rh = std::min({
                            gridCenterY,
                            gridCenterY2,
                            plainHeight[0] - gridCenterY,
                            plainHeight[1] - gridCenterY2,
                        });

                        if (rw < 1) {
                            continue;
                        }

                        if (rh < 1) {
                            continue;
                        }

                        const std::vector<cv::Point2f> src = {
                                {(float)plainImageGrid[0][(gridCenterY - rh) * plainWidth[0] + gridCenterX - rw].x, (float)plainImageGrid[0][(gridCenterY - rh) * plainWidth[0] + gridCenterX - rw].y},
                                {(float)plainImageGrid[0][(gridCenterY - rh) * plainWidth[0] + gridCenterX + rw - 1].x, (float)plainImageGrid[0][(gridCenterY - rh) * plainWidth[0] + gridCenterX + rw - 1].y},
                                {(float)plainImageGrid[0][(gridCenterY + rh - 1) * plainWidth[0] + gridCenterX - rw].x, (float)plainImageGrid[0][(gridCenterY + rh - 1) * plainWidth[0] + gridCenterX - rw].y},
                                {(float)plainImageGrid[0][(gridCenterY + rh - 1) * plainWidth[0] + gridCenterX + rw - 1].x, (float)plainImageGrid[0][(gridCenterY + rh - 1) * plainWidth[0] + gridCenterX + rw - 1].y},
                        };
                        const std::vector<cv::Point2f> dest = {
                                {(float)plainImageGrid[1][(gridCenterY2 - rh) * plainWidth[1] + gridCenterX2 - rw].x, (float)plainImageGrid[1][(gridCenterY2 - rh) * plainWidth[0] + gridCenterX2 - rw].y},
                                {(float)plainImageGrid[1][(gridCenterY2 - rh) * plainWidth[1] + gridCenterX2 + rw - 1].x, (float)plainImageGrid[1][(gridCenterY2 - rh) * plainWidth[0] + gridCenterX2 + rw - 1].y},
                                {(float)plainImageGrid[1][(gridCenterY2 + rh - 1) * plainWidth[1] + gridCenterX2 - rw].x, (float)plainImageGrid[1][(gridCenterY2 + rh - 1) * plainWidth[0] + gridCenterX2 - rw].y},
                                {(float)plainImageGrid[1][(gridCenterY2 + rh - 1) * plainWidth[1] + gridCenterX2 + rw - 1].x, (float)plainImageGrid[1][(gridCenterY2 + rh - 1) * plainWidth[0] + gridCenterX2 + rw - 1].y},
                        };
                        transform = cv::getPerspectiveTransform(src, dest);
                    }

                    CV_Assert(transform.type() == CV_64FC1);

                    bestMap1[1].copyTo(alignedMap);

                    for (auto it = alignedMap.begin<Point2f>(), end = alignedMap.end<Point2f>(); it != end; it++) {
                        auto p = *it;
                        std::vector<double> _p = {p.x, p.y, 1};
                        cv::Mat _p2(transform.size().height, 1, CV_64FC1);
                        auto p2 = (cv::Point3d *)_p2.data;
                        cv::gemm(transform, Mat(1, 3, CV_64FC1, _p.data()), 1, Mat(), 0, _p2, cv::GemmFlags::GEMM_2_T);
                        *it = {(float)(p2->x / p2->z), (float)(p2->y / p2->z)};
                    }

                    cv::remap(frames[1], aligned, alignedMap, noArray(), INTER_NEAREST);
                    imshow("Aligned", aligned);

                    for (int i = 0; i < frames.size(); ++i) {
                        invMap(bestMap1[i], invBestMap1[i]);
                    }
                    invMap(alignedMap, invAlignedMap);

                    matwrite("map0.bin", bestMap1[0]);
                    matwrite("map1.bin", bestMap1[1]);
                    matwrite("mapa.bin", alignedMap);
                }
            }
        } else {
            std::vector<cv::Mat> result(frames.size());
            for (int j = 0; j < frames.size(); ++j) {
                for (int k = 1; k < 12; ++k) {
                    auto p1 = Point(0, k * frames[j].size().height / 11);
                    auto p2 = Point(frames[j].size().width, k * frames[j].size().height / 11);
                    cv::line(frames[j], p1, p2, cv::Scalar(255, 0, 255), 1);
                }
                if (j != 1) {
                    cv::remap(frames[j], result[j], bestMap1[j], cv::Mat(), INTER_NEAREST);
                } else {
                    cv::remap(frames[1], result[j], alignedMap, noArray(), INTER_NEAREST);
                }

                for (int k = 1; k < 12; ++k) {
                    auto p1 = Point(0, k * frames[j].size().height / 11);
                    auto p2 = Point(frames[j].size().width, k * frames[j].size().height / 11);
                    cv::line(result[j], p1, p2, cv::Scalar(0, 255, 255), 1);
                }

            }

            cv::drawMarker(frames[0], mouseSrc, cv::Scalar(255, 0, 255), MarkerTypes::MARKER_CROSS, 30, 2);
            auto plainMap1 = invBestMap1[0].ptr<cv::Point2f>(mouseSrc.y, mouseSrc.x);
            auto plainMap2 = alignedMap.ptr<cv::Point2f>((int)plainMap1->y, (int)plainMap1->x);
            cv::drawMarker(result[0], cv::Point2i((int) plainMap1->x, (int) plainMap1->y), cv::Scalar(0, 255, 0), MarkerTypes::MARKER_TILTED_CROSS, 30, 2);
            cv::drawMarker(frames[1], cv::Point2i((int) plainMap2->x, (int) plainMap2->y), cv::Scalar(0, 255, 0), MarkerTypes::MARKER_CROSS, 30, 2);
            cv::drawMarker(result[1], cv::Point2i((int) plainMap1->x, (int) plainMap1->y), cv::Scalar(0, 255, 0), MarkerTypes::MARKER_TILTED_CROSS, 30, 2);


            for (int j = 0; j < frames.size(); ++j) {
                imshow("Plain best " + std::to_string(j), result[j]);
                imshow("src " + std::to_string(j), frames[j]);
            }

//            for (int j = 1; j < 12; ++j) {
//                auto p1 = Point(0, j * aligned.size().height / 11);
//                auto p2 = Point(aligned.size().width, j * aligned.size().height / 11);
//                cv::line(aligned, p1, p2, cv::Scalar(0, 255, 255), 1);
//            }

//            cv::drawMarker(aligned, mousePlain, cv::Scalar(255, 0, 255), MarkerTypes::MARKER_TILTED_CROSS, 30, 2);
//            imshow("Aligned", aligned);
        }

//        processor.processFrame(inputLeft, inputRight, output);
//
//        cv::resize(inputLeft, outputLeft, outputSize, 0, 0,
//                   cv::INTER_NEAREST);
//
//        cv::resize(inputRight, outputRight, outputSize, 0, 0,
//                   cv::INTER_NEAREST);

        //drawText(outputLeft, calibrationRMS);

       // outputLeft.copyTo(resultLeft);
       // outputRight.copyTo(resultRight);

        auto end = std::chrono::high_resolution_clock::now();

        double time = ((double) (end - start).count()) / 1e6;
        double avgA = 2. / ((i < 50 ? 10 : 100) + 1);
        avgTime = avgTime == 0. ? time : avgA * time + (1 - avgA) * avgTime;
        avgFps = avgFps == 0. ? fps : avgA * fps + (1 - avgA) * avgFps;

        std::cerr << "fps " << avgFps << " time " << time << " load " << avgTime * 100 / ((double)us / 1e6) << " %, avg " << avgTime << " size "
                  << resultLeft.dataend - resultLeft.datastart << std::endl;

        std::ostringstream tm;

        tm << "PERF " << fps << " " << time << std::endl;
//
//        if (!writerIsRunning) {
//            if (writer.joinable()) {
//                writer.join();
//            }
//
//            writerIsRunning = true;
//
//            writer = std::thread([](auto *resultLeft, auto *pipe, auto *isRunning, auto *tm) {
//                broadcastingServer.broadcast(tm->str());
//
//                fwrite(resultLeft->data, sizeof(char), resultLeft->dataend - resultLeft->datastart, *pipe);
//                *isRunning = false;
//            }, &resultLeft, &pipe, &writerIsRunning, &tm);
//        }

#ifdef HAVE_OPENCV_HIGHGUI
//        imshow("Left", resultLeft);
//        if (!resultRight.empty()) {
//            imshow("Right", resultRight);
//        }

        if (cv::waitKey(1) >= 0) {
            needsCalibration1 = !needsCalibration1;
        }
#endif
    }

//    onlineCalibration.storeCalibrationImages();
//    onlineCalibration.storeCalibrationResult();

    if (writer.joinable()) {
        writer.join();
    }

    readerLeft.join();
    readerRight.join();
    if (calibration.joinable()) {
        calibration.join();
    }

    std::cerr << "exiting..." << std::endl;

    captureLeft.release();
    captureRight.release();

#ifdef HAVE_OPENCV_HIGHGUI
    cv::destroyAllWindows();
#endif

    broadcastingServerThread.join();
    commandServerThread.join();

    return 0;
}

int findNearestPoint(const Point2f &point, const std::vector<Point3d> &points, double searchRadius) {
    int found = -1;
    auto foundDistance = 1e6;
    for (int i = 0; i < points.size(); i++) {
        auto p = points[i];
        if (std::abs(p.x - point.x) > searchRadius || std::abs(p.y - point.y) > searchRadius) {
            continue;
        }

        auto d = std::sqrt(std::pow(p.x - point.x, 2) + std::pow(p.y - point.y, 2));

        if (d < foundDistance) {
            foundDistance = d;
            found = i;
        }
    }

    return found;
}

void drawText(UMat &image, double rms) {
    char text[100];

    snprintf(text, 100, "Hello, Opencv. RMS %f", rms);

    cv::putText(image, text,
                cv::Point(20, 50),
                cv::FONT_HERSHEY_COMPLEX, 1, // font face and scale
                cv::Scalar(255, 255, 255), // white
                1, cv::LINE_AA); // line thickness and type
}
