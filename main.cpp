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

#include "QuickStereoMatch.h"
#include "CalibrateMapper.h"

using namespace mini_server;

void drawText(UMat &image, double rms);

static std::atomic running = true;

static BroadcastingServer broadcastingServer;
static CommandServer commandServer;

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

    int patternSize = 60;
    cv::Mat map1, map2;

    bool needsCalibration1 = true;

    std::vector<double> avgDistCoeff(14);
    bool avgDistCoeffInitialized = false;
    double prevRmse = 1000;
    double prevStdRmse = 1000;
    double prevRmseUndistorted = 1000;
    double prevStdRmseUndistorted = 1000;
    size_t nextFrameId = 0;

    ecv::CalibrateMapper<double> calibrateMapper;

    std::vector<std::vector<Point3f>> objectPoints(10);
    std::vector<std::vector<Point2f>> imagePoints(10);

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
        cv::rotate(readingImRight, imageRight, ROTATE_180);
        readerRightLock.unlock();

        auto now = std::chrono::high_resolution_clock::now();
        auto us = (double) (now - prev).count();
        prev = now;

        fps = 1e9 / us;

        if (imageLeft.empty()) {
            break;
        }

        if (needsCalibration1) {
            Mat colorLeft = imageLeft.getMat(AccessFlag::ACCESS_RW);
            calibrateMapper.setPatternSize(patternSize);
            std::vector<cv::Point3d> peaks(1000);
            std::vector<Point3d> imageGrid(peaks.size());
            std::vector<Point3d> objectGrid(peaks.size());
            size_t gridWidth = 0;
            size_t gridHeight = 0;
            size_t size = peaks.size();
            ecv::CalibrateMapper<double>::BaseSquare square;

            calibrateMapper.detectPeaks(colorLeft, peaks, &size);
            auto squareRms = calibrateMapper.detectBaseSquare(colorLeft.size(), peaks, square);
            auto gridRmse = calibrateMapper.detectFrameImagePointsGrid(colorLeft.size(), peaks, square, imageGrid, &gridWidth, &gridHeight);

            if (squareRms < 0.05f && !std::isnan(squareRms) && (gridWidth * gridHeight) > 0) {
                auto list = {
                    square.topRight.x - square.topLeft.x,
                    square.bottomLeft.y - square.topLeft.y,

                    (imageGrid[1].x - imageGrid[0].x),
                    (imageGrid[gridWidth - 1].x - imageGrid[gridWidth - 2].x),
                    (imageGrid[(gridHeight - 1) * gridWidth + 1].x - imageGrid[(gridHeight - 1) * gridWidth].x),
                    (imageGrid[(gridHeight - 1) * gridWidth + gridWidth - 1].x - imageGrid[(gridHeight - 1) * gridWidth + gridWidth - 2].x),

                    (imageGrid[1 * gridWidth].y - imageGrid[0].y),
                    (imageGrid[1 * gridWidth + gridWidth - 1].y - imageGrid[gridWidth - 1].y),

                    (imageGrid[(gridHeight - 1) * gridWidth].y - imageGrid[(gridHeight - 2) * gridWidth].y),
                    (imageGrid[(gridHeight - 1) * gridWidth + gridWidth - 1].y - imageGrid[(gridHeight - 2) * gridWidth + gridWidth - 1].y),
                };
                patternSize = (int)std::min(list);
                patternSize -= patternSize % 2;
                patternSize = std::min(256, std::max(24, patternSize));
            } else {
                patternSize = (int)(random() % (65 - 24) + 24);
                patternSize -= patternSize % 2;
            }

            auto stdGridRmse = calibrateMapper.generateFrameObjectPointsGrid(imageGrid, objectGrid, gridWidth, gridHeight);

            calibrateMapper.drawPeaks(colorLeft, peaks, size, cv::Scalar(0, 255, 0));
            calibrateMapper.drawBaseSquare(colorLeft, square, squareRms < 0.1f ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255));
            calibrateMapper.drawGrid(colorLeft, imageGrid, gridWidth, gridHeight, cv::Scalar(255, 0, 0));
            calibrateMapper.drawGrid(colorLeft, objectGrid, gridWidth, gridHeight, cv::Scalar(255, 255, 0));
            calibrateMapper.drawGridCorrelation(colorLeft, imageGrid, objectGrid, gridWidth, gridHeight, cv::Scalar(255, 0, 255));

            std::cerr << "Peaks " << size << "; square q " << squareRms << "; stdGridRmse " << stdGridRmse << "; stdGridRmse " << patternSize << std::endl;

            if (
                calibrateMapper.isGridValid(colorLeft.size(), imageGrid, gridWidth, gridHeight)
                && calibrateMapper.isGridValid(colorLeft.size(), objectGrid, gridWidth, gridHeight)
                && gridRmse < 100 && stdGridRmse < 50
            ) {
                auto frameId = nextFrameId % objectPoints.size();
                nextFrameId++;

                imagePoints[frameId].resize(gridWidth * gridHeight);
                objectPoints[frameId].resize(gridWidth * gridHeight);

                calibrateMapper.convertTo2dPoints(imageGrid, imagePoints[frameId]);
                calibrateMapper.convertToPlain3dPoints(objectGrid, objectPoints[frameId]);

                Mat cameraMatrix = Mat::zeros(3, 3, CV_64F);
                std::vector<double> distCoeff(14);
                Mat rvecs = Mat::zeros(3, 3, CV_64F);
                Mat tvecs = Mat::zeros(3, 3, CV_64F);

                cameraMatrix.setTo(Scalar::all(0));
                auto cameraData = (double *)cameraMatrix.data;

                cameraData[0] = 8000.;
                cameraData[4] = 8000.;
                cameraData[8] = 1.;

                map1.setTo(Scalar::all(0));
                map2.setTo(Scalar::all(0));

                int baseFlags =
                        CALIB_USE_EXTRINSIC_GUESS|
                        CALIB_FIX_PRINCIPAL_POINT|
                        CALIB_RATIONAL_MODEL|
                        CALIB_THIN_PRISM_MODEL|
                        CALIB_TILTED_MODEL;

                std::vector<int> flagsChain = {
//                        CALIB_USE_INTRINSIC_GUESS|
//                        CALIB_FIX_FOCAL_LENGTH|
                        CALIB_FIX_TAUX_TAUY|
                        CALIB_FIX_TANGENT_DIST|
                        CALIB_FIX_S1_S2_S3_S4|
//                        CALIB_FIX_K1|
//                        CALIB_FIX_K2|
//                        CALIB_FIX_K3|
                        CALIB_FIX_K4|
                        CALIB_FIX_K5|
                        CALIB_FIX_K6|
                        CALIB_FIX_ASPECT_RATIO,

                        CALIB_USE_INTRINSIC_GUESS|
                        CALIB_FIX_FOCAL_LENGTH|
                        CALIB_FIX_TAUX_TAUY|
                        CALIB_FIX_TANGENT_DIST|
//                        CALIB_FIX_S1_S2_S3_S4|
                        CALIB_FIX_K1|
                        CALIB_FIX_K2|
//                        CALIB_FIX_K3|
//                        CALIB_FIX_K4|
//                        CALIB_FIX_K5|
//                        CALIB_FIX_K6|
                        CALIB_FIX_ASPECT_RATIO,

                        CALIB_USE_INTRINSIC_GUESS|
                        CALIB_FIX_FOCAL_LENGTH|
                        CALIB_FIX_TAUX_TAUY|
//                        CALIB_FIX_TANGENT_DIST|
                        CALIB_FIX_S1_S2_S3_S4|
                        CALIB_FIX_K1|
                        CALIB_FIX_K2|
                        CALIB_FIX_K3|
                        CALIB_FIX_K4|
                        CALIB_FIX_K5|
                        CALIB_FIX_K6|
                        CALIB_FIX_ASPECT_RATIO,
                    };

                if (nextFrameId > objectPoints.size()) {

                    try {
                        for (auto flags: flagsChain) {
                            cv::calibrateCamera(objectPoints, imagePoints, colorLeft.size(), cameraMatrix,
                                                distCoeff, rvecs, tvecs, baseFlags | flags,
                                                TermCriteria(10000, 0.01)
                            );
                        }
                    } catch (const std::exception &e) {
                        std::cerr << "cv::calibrateCamera failed" << std::endl;
                    }

                    cv::initUndistortRectifyMap(cameraMatrix, distCoeff, noArray(), cameraMatrix, colorLeft.size(),
                                                CV_32FC2,
                                                map1, map2);
                }
            }

            if (!map1.empty()) {
                Mat plainLeft;
                cv::remap(readingImLeft, plainLeft, map1, map2, INTER_LINEAR);


                imshow("Plain", plainLeft);
            }
            imshow("GL2", colorLeft);

            imageLeft.copyTo(inputLeft);
            imageRight.copyTo(inputRight);
            imageLeft.copyTo(outputLeft);
            imageRight.copyTo(outputRight);
        } else {
            cv::remap(imageLeft, resultLeft, map1, map2, INTER_LINEAR);
            imageRight.copyTo(inputRight);
            imageLeft.copyTo(outputLeft);
            imageRight.copyTo(outputRight);
            imshow("Plain", resultLeft);
        }

//        processor.processFrame(inputLeft, inputRight, output);
//
//        cv::resize(inputLeft, outputLeft, outputSize, 0, 0,
//                   cv::INTER_NEAREST);
//
//        cv::resize(inputRight, outputRight, outputSize, 0, 0,
//                   cv::INTER_NEAREST);

        drawText(outputLeft, calibrationRMS);

        outputLeft.copyTo(resultLeft);
        outputRight.copyTo(resultRight);

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

void drawText(UMat &image, double rms) {
    char text[100];

    snprintf(text, 100, "Hello, Opencv. RMS %f", rms);

    cv::putText(image, text,
                cv::Point(20, 50),
                cv::FONT_HERSHEY_COMPLEX, 1, // font face and scale
                cv::Scalar(255, 255, 255), // white
                1, cv::LINE_AA); // line thickness and type
}
