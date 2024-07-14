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

#include <DispEst.h>
#include "QuickStereoMatch.h"

using namespace mini_server;

void drawText(UMat &image, double rms, unsigned long i);

bool isCalibrationComplete(const UMat *mat1, const UMat *mat2, const StereoCameraProperties &props);

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
        cv::VideoCaptureProperties::CAP_PROP_ORIENTATION_AUTO, false,
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

    OnlineCalibration onlineCalibration;

    auto needsCalibration = !onlineCalibration.loadCalibrationResult();

    std::atomic calibrating = false;
    std::atomic<double> calibrationRMS = INFINITY;
    std::thread calibration;

    UMat calibLeft, calibRight;

    StereoCameraProperties &props = onlineCalibration.props;

    onlineCalibration.setHighPrecision();

    if (needsCalibration) {
        onlineCalibration.loadCalibrationImages();
        calibrationRMS = onlineCalibration.updateCalibrationResult();
        onlineCalibration.rectify();
    }

    DispEst *SMDE = nullptr;

    // Use create ROI which is valid in both left and right ROIs
    int tl_x = MAX(props.roi[0].x, props.roi[1].x);
    int tl_y = MAX(props.roi[0].y, props.roi[1].y);
    int br_x = MIN(props.roi[0].width + props.roi[0].x, props.roi[1].width + props.roi[1].x);
    int br_y = MIN(props.roi[0].height + props.roi[0].y, props.roi[1].height + props.roi[1].y);
    auto cropBox = Rect(Point(tl_x, tl_y), Point(br_x, br_y));

    UMat lFrame, rFrame;
    UMat lDispMap, rDispMap, leftDispMap, rightDispMap;

    QuickStereoMatch sm;
//
//    Mat ltest = (Mat_<unsigned char>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, })).reshape(1);
//    Mat rtest = (Mat_<unsigned char>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, })).reshape(1);
//    Mat ldtest;
//    Mat rdtest;
//
//    sm.computeDisparityMap(ltest, rtest, ldtest, rdtest, 5, 6, 5);
//
//    std::cout << ldtest.rowRange(5, 6) << std::endl << std::endl;
//    std::cout << rdtest.rowRange(5, 6) << std::endl << std::endl;

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

        auto now = std::chrono::high_resolution_clock::now();
        auto us = (double) (now - prev).count();
        prev = now;

        fps = 1e9 / us;

        if (imageLeft.empty()) {
            break;
        }

        if (needsCalibration && i % 30 == 0 && !calibrating) {
            if (onlineCalibration.autoDetectBoard(imageLeft, imageRight)) {
                calibrating = true;

                imageLeft.copyTo(calibLeft);
                imageRight.copyTo(calibRight);

                if (calibration.joinable()) {
                    calibration.join();
                }

                calibration = std::thread([](OnlineCalibration *onlineCalibration, auto *calibrating, UMat *imageLeft, UMat *imageRight, StereoCameraProperties *props, auto *calibrationRMS) {
                    onlineCalibration->drawChessboardCorners(*imageLeft, *imageRight);

                    cv::UMat left, right;

                    cv::rectangle(*imageLeft, props->roi[0], cv::Scalar(94, 206, 165), 8);
                    cv::rectangle(*imageRight, props->roi[1], cv::Scalar(94, 206, 165), 8);

                    cv::resize(*imageLeft, left, cv::Size(0, 0), 0.25, 0.25);
                    cv::resize(*imageRight, right, cv::Size(0, 0), 0.25, 0.25);

                    cv::imshow("Calibration Left", left);
                    cv::imshow("Calibration Right", right);
                    cv::waitKey(1);
                    auto rms = onlineCalibration->updateCalibrationResult();
                    if (rms > 10) {
                        onlineCalibration->rejectLastBoard();
                    } else {
                        *calibrationRMS = rms;

                        if (onlineCalibration->imagePointsL.size() > 10) {
                            onlineCalibration->rectify();
                            onlineCalibration->storeCalibrationResult();
                        }
                    }

                    cv::rectangle(*imageLeft, props->roi[0], cv::Scalar(94, 206, 165), 8);
                    cv::rectangle(*imageRight, props->roi[1], cv::Scalar(94, 206, 165), 8);

                    cv::resize(*imageLeft, left, cv::Size(0, 0), 0.25, 0.25);
                    cv::resize(*imageRight, right, cv::Size(0, 0), 0.25, 0.25);

                    cv::imshow("Calibration Left", left);
                    cv::imshow("Calibration Right", right);
                    cv::waitKey(1);
                    *calibrating = false;
                }, &onlineCalibration, &calibrating, &calibLeft, &calibRight, &props, &calibrationRMS);
            }
        }

        if (onlineCalibration.isCalibrated) {
            remap(imageLeft, inputLeft, onlineCalibration.rmapL[0], onlineCalibration.rmapL[1], INTER_LINEAR);
            remap(imageRight, inputRight, onlineCalibration.rmapR[0], onlineCalibration.rmapR[1], INTER_LINEAR);

            inputLeft = inputLeft(cropBox);
            auto cropBox2 = Rect(Point(tl_x + 4, tl_y), Point(br_x + 4, br_y));
            inputRight = inputRight(cropBox2);

            UMat inputLeft2, inputRight2;

//            cv::edgePreservingFilter(inputLeft, inputLeft2);
//            cv::edgePreservingFilter(inputRight, inputRight2);
//            cv::bilateralFilter(inputLeft, inputLeft2, 15, 75, 75);
//            cv::bilateralFilter(inputRight, inputRight2, 15, 75, 75);
//            fastGuidedFilter(inputLeft, inputLeft, 70, 0.001);
//            fastGuidedFilter(inputRight, inputRight, 70, 0.001);
//            inputLeft2.copyTo(inputLeft);
//            inputRight2.copyTo(inputRight);
//            inputLeft.copyTo(inputLeft2);
//            inputRight.copyTo(inputRight2);

            cv::rotate(inputLeft, inputLeft, ROTATE_180);
            cv::rotate(inputRight, inputRight, ROTATE_180);

            cv::rotate(inputLeft, outputLeft, ROTATE_90_CLOCKWISE);
            cv::rotate(inputRight, outputRight, ROTATE_90_CLOCKWISE);
//            cv::resize(outputLeft, outputLeft, Size(0, 0), 0.25, 0.25);
//            cv::resize(outputRight, outputRight, Size(0, 0), 0.25, 0.25);
//
//            auto kernel = (Mat_<float>(5, 5) <<
//                    0, 2,  30,  2,  0,
//                    0, 5,  70, 5,  0,
//                    1, 10, 100, 10, 1,
//                    0, 5,  70, 5,  0,
//                    0, 2,  30,  2,  0
//                          ) / 1000;
//
//            cv::filter2D(outputLeft, outputLeft, kernel);
//            cv::filter2D(outputRight, outputRight, kernel);

            UMat map2, map3;

            std::vector<Mat> leftPyramid, rightPyramid, lmap, rmap;

            cv::buildPyramid(outputLeft, leftPyramid, 4);
            cv::buildPyramid(outputRight, rightPyramid, 4);

            double max = leftPyramid[3].cols * 0.45;

            lmap.resize(5);
            rmap.resize(5);

            for (int j = 0; j < 4; ++j) {
                lmap[j] = Mat::zeros(lmap[j].size(), CV_32FC1);
                rmap[j] = Mat::zeros(rmap[j].size(), CV_32FC1);
            }

            int borderSize = 10;
            sm.computeDisparityMap(leftPyramid[3], rightPyramid[3], lmap[3], rmap[3], (int)max, 10, borderSize);

            int scale = 3;

            for (int j = 2; j >= scale; --j) {
                cv::medianBlur(lmap[j + 1], lmap[j + 1], 3);
                cv::resize(lmap[j + 1], lmap[j], Size(0, 0), 2, 2);
                cv::resize(rmap[j + 1], rmap[j], Size(0, 0), 2, 2);
                lmap[j] *= 2;
                rmap[j] *= 2;
                max *= 2;
                borderSize *= 2;
                sm.computeDisparityMap(leftPyramid[j], rightPyramid[j], lmap[j], rmap[j], 10, 15, borderSize);
            }

            rmap[scale].copyTo(map2);

            Mat m;
            double min = -max;
            map2.convertTo(m, CV_8UC1, 255 / (max-min), -min);
//            cv::resize(m, m, Size(0, 0), 2, 2);
            cv::rotate(m, m, ROTATE_90_COUNTERCLOCKWISE);

            cv::Mat falseColorsMap;
            applyColorMap(m, falseColorsMap, cv::COLORMAP_TURBO);

            imshow("DispMap R", falseColorsMap);

            lmap[scale].copyTo(map2);
//            cv::medianBlur(lmap, map2, 3);
//            cv::medianBlur(map2, map2, 3);
//            cv::medianBlur(map2, map2, 3);
//            cv::medianBlur(map2, map2, 3);

            map2.convertTo(m, CV_8UC1, 255 / (max-min), -min);
//            cv::resize(m, m, Size(0, 0), 2, 2);
            cv::rotate(m, m, ROTATE_90_COUNTERCLOCKWISE);
            cv::Mat falseColorsMap2;
            applyColorMap(m, falseColorsMap2, cv::COLORMAP_TURBO);

            imshow("DispMap L", falseColorsMap2);

//            fastGuidedFilter(inputLeft, inputLeft, 32, 0.0001f, 8);

//            if (SMDE == nullptr) {
//                SMDE = new DispEst(inputLeft, inputRight, 50, 20, true);
//            }
//
//            inputLeft.convertTo(lFrame, CV_32F, 1 / 255.0f);
//            inputRight.convertTo(rFrame, CV_32F,  1 / 255.0f);
//
//            SMDE->setInputImages(lFrame, lFrame);
//            SMDE->setSubsampleRate(3);

//            SMDE->CostConst_GPU();
//            cv::waitKey(1);
//            SMDE->CostFilter_FGF();
//            cv::waitKey(1);
//            SMDE->DispSelect_GPU();
//            cv::waitKey(1);
//            SMDE->PostProcess_GPU();
//            cv::waitKey(1);

//            // ******** Display Disparity Maps  ******** //
//            SMDE->lDisMap.copyTo(lDispMap); //scale factor used to compare error with ground truth
//            SMDE->rDisMap.copyTo(rDispMap); //scale factor used to compare error with ground truth
//
//            cv::cvtColor(lDispMap, leftDispMap, cv::COLOR_GRAY2RGB);
//            cv::cvtColor(lDispMap, rightDispMap, cv::COLOR_GRAY2RGB);
//
//            imshow("Left DispMap", lDispMap);
//            imshow("Right DispMap", rDispMap);

//            imageLeft.copyTo(inputLeft);
//            imageRight.copyTo(inputRight);
        } else {
            imageLeft.copyTo(inputLeft);
            imageRight.copyTo(inputRight);
        }

//        processor.processFrame(inputLeft, inputRight, output);

        cv::resize(inputLeft, outputLeft, outputSize, 0, 0,
                   cv::INTER_NEAREST);

        cv::resize(inputRight, outputRight, outputSize, 0, 0,
                   cv::INTER_NEAREST);

        drawText(outputLeft, calibrationRMS, onlineCalibration.imagePointsL.size());

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
        imshow("Left", resultLeft);
        if (!resultRight.empty()) {
            imshow("Right", resultRight);
        }

        if (cv::waitKey(1) >= 0) {
            running = false;
        }
#endif
    }

    onlineCalibration.storeCalibrationImages();
    onlineCalibration.storeCalibrationResult();

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

bool isCalibrationComplete(const UMat *mat1, const UMat *mat2, const StereoCameraProperties &props) {
    return props.roi[0].height > 0.8 * mat1->rows && props.roi[0].width > 0.8 * mat1->cols
        && props.roi[1].height > 0.8 * mat2->rows && props.roi[1].width > 0.8 * mat2->cols;
}

void drawText(UMat &image, double rms, unsigned long i) {
    char text[100];

    snprintf(text, 100, "Hello, Opencv. RMS %f. N %lu", rms, i);

    cv::putText(image, text,
                cv::Point(20, 50),
                cv::FONT_HERSHEY_COMPLEX, 1, // font face and scale
                cv::Scalar(255, 255, 255), // white
                1, cv::LINE_AA); // line thickness and type
}
