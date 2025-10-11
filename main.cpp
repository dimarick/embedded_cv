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

using namespace mini_server;

void drawText(UMat &image, double rms, unsigned long i);

void findPeaks(const Mat &mat, std::vector<cv::Point3f> &points, size_t *size, int kernel, int noiseTolerance = 2);
void createCheckboadPatterns(Mat &t1);

cv::Point3f findMassCenter(const Mat &mat, int x, int y, int searchRadius);

struct CenterQuad {
    cv::Point3f topLeft;
    cv::Point3f topRight;
    cv::Point3f bottomLeft;
    cv::Point3f bottomRight;
};
float quadQualityNormRms(CenterQuad quad);
float quadQualityRms(CenterQuad quad);
float findCenterQuad(Size size, const std::vector<cv::Point3f> &points, CenterQuad &result);

float findGrid(
        const Size &size,
        const CenterQuad &quad,
        std::vector<cv::Point3f> &points,
        std::vector<Point3f> &grid,
        size_t *gridWidth,
        size_t *gridHeight
);

Point3f findNearestPoint(const Point3f &point3, std::vector<cv::Point3f> &vector, float d);

void fillGridRow(size_t w, int cH, int cW, int j, std::vector<cv::Point3f> &points, std::vector<Point3f> &grid);

void cropGrid(std::vector<Point3f> &grid, size_t *w, size_t *h);

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
//
//    DispEst *SMDE = nullptr;

    // Use create ROI which is valid in both left and right ROIs
    int tl_x = MAX(props.roi[0].x, props.roi[1].x);
    int tl_y = MAX(props.roi[0].y, props.roi[1].y);
    int br_x = MIN(props.roi[0].width + props.roi[0].x, props.roi[1].width + props.roi[1].x);
    int br_y = MIN(props.roi[0].height + props.roi[0].y, props.roi[1].height + props.roi[1].y);
    auto cropBox = Rect(Point(tl_x, tl_y), Point(br_x, br_y));

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

    int patternSize = 64;
    auto t1 = Mat(Size(patternSize + 1, patternSize + 1), CV_8U);
    createCheckboadPatterns(t1);
    auto t2 = 1 - t1;
    std::vector<cv::Point3f> points(1000);

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

        if (true) {
            Mat grayLeft;
            Mat colorLeft = imageLeft.getMat(AccessFlag::ACCESS_RW);
            cv::cvtColor(imageLeft, grayLeft, COLOR_BGR2GRAY);
            cv::equalizeHist(grayLeft, grayLeft);
            auto leftMatches1 = Mat(grayLeft.size(), CV_32F);
            auto leftMatches2 = Mat(grayLeft.size(), CV_32F);
            cv::matchTemplate(grayLeft, t1, leftMatches1, TemplateMatchModes::TM_CCOEFF_NORMED, noArray());
            cv::multiply(leftMatches1, leftMatches1, leftMatches1);
            cv::copyMakeBorder(leftMatches1,leftMatches1, patternSize/2, patternSize/2, patternSize/2, patternSize/2, BorderTypes::BORDER_REPLICATE);
            size_t size = points.size();

            findPeaks(leftMatches1, points, &size, patternSize / 2, 2);

            std::sort(points.begin(), points.begin() + (int)size - 1, [](Point3f a, Point3f b) {
                return a.z > b.z;
            });

            CenterQuad quad;
            auto quadRms = findCenterQuad(leftMatches1.size(), points, quad);

            float gridRms = 0.0;
            std::vector<Point3f> grid(points.size());
            size_t gridWidth;
            size_t gridHeight;
            if (quadRms < 0.1) {
                gridRms = findGrid(leftMatches1.size(), quad, points, grid, &gridWidth, &gridHeight);
            }

            auto center = Point((int)colorLeft.size().width / 2, (int)colorLeft.size().height / 2);
            cv::drawMarker(colorLeft, center, cv::Scalar(255, 0, 0), MARKER_DIAMOND, 20, 2);

            for (int j = 0; j < size; ++j) {
                auto p = points[j];
                if (p.z <= 0) {
                    continue;
                }
                if (p.x <= 0) {
                    continue;
                }

                if (p.y <= 0) {
                    continue;
                }

                char info[256];
                snprintf(info, sizeof info - sizeof("\0"), "%d", j);

                cv::drawMarker(leftMatches1, Point((int)p.x, (int)p.y), cv::Scalar(0, 255, 0), MARKER_TILTED_CROSS, 50 * p.z, 2);
                cv::drawMarker(colorLeft, Point((int)p.x, (int)p.y), cv::Scalar(0, 255, 0), MARKER_TILTED_CROSS, 50 * p.z, 2);
                cv::putText(colorLeft, info, Point((int)p.x, (int)p.y), FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
            }

            auto color = quadRms < 0.1f ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
            cv::line(colorLeft, Point((int) quad.topLeft.x, (int) quad.topLeft.y),
                     Point((int) quad.topRight.x, (int) quad.topRight.y), color, 3);
            cv::line(colorLeft, Point((int) quad.topLeft.x, (int) quad.topLeft.y),
                     Point((int) quad.bottomLeft.x, (int) quad.bottomLeft.y), color, 3);
            cv::line(colorLeft, Point((int) quad.topLeft.x, (int) quad.topLeft.y),
                     Point((int) quad.bottomRight.x, (int) quad.bottomRight.y), color, 3);
            cv::line(colorLeft, Point((int) quad.topRight.x, (int) quad.topRight.y),
                     Point((int) quad.bottomLeft.x, (int) quad.bottomLeft.y), color, 3);
            cv::line(colorLeft, Point((int) quad.topRight.x, (int) quad.topRight.y),
                     Point((int) quad.bottomRight.x, (int) quad.bottomRight.y), color, 3);
            cv::line(colorLeft, Point((int) quad.bottomLeft.x, (int) quad.bottomLeft.y),
                     Point((int) quad.bottomRight.x, (int) quad.bottomRight.y), color, 3);

            for (int x = 0; x < gridWidth; ++x) {
                for (int y = 0; y < gridHeight; ++y) {
                    auto p = grid[y * gridWidth + x];
                    auto right = grid[y * gridWidth + x + 1];
                    auto bottom = grid[(y + 1) * gridWidth + x];

                    if (p.z < 0) {
                        continue;
                    }

                    if (right.z > 0 && x < gridWidth - 1) {
                        cv::line(colorLeft, Point((int) p.x, (int) p.y), Point((int) right.x, (int) right.y),
                                 cv::Scalar(255, 0, 0), 5);
                    }

                    if (bottom.z > 0 && y < gridHeight - 1) {
                        cv::line(colorLeft, Point((int) p.x, (int) p.y), Point((int) bottom.x, (int) bottom.y),
                                 cv::Scalar(255, 0, 0), 5);
                    }
                }
            }

            std::cerr << "Peaks " << size << "; square q " << quadRms << std::endl;

            imshow("GL1", leftMatches1);
            imshow("GL2", colorLeft);

            imageLeft.copyTo(inputLeft);
            imageRight.copyTo(inputRight);
            imageLeft.copyTo(outputLeft);
            imageRight.copyTo(outputRight);
        } else {
            imageLeft.copyTo(inputLeft);
            imageRight.copyTo(inputRight);
            imageLeft.copyTo(outputLeft);
            imageRight.copyTo(outputRight);
        }

//        processor.processFrame(inputLeft, inputRight, output);
//
//        cv::resize(inputLeft, outputLeft, outputSize, 0, 0,
//                   cv::INTER_NEAREST);
//
//        cv::resize(inputRight, outputRight, outputSize, 0, 0,
//                   cv::INTER_NEAREST);

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
//        imshow("Left", resultLeft);
//        if (!resultRight.empty()) {
//            imshow("Right", resultRight);
//        }

        if (cv::waitKey(1) >= 0) {
            running = false;
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

float distance(Point3f p1, Point3f p2) {
    return (float)std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

Point3f approximate(Point3f current, Point3f prev) {
    return {current.x + (current.x - prev.x), current.y + (current.y - prev.y), (current.z + prev.z) / 2};
}

/**
 * Находит вероятное расположение следующей (слева-сверху) точки, по текущей, точке слева и точке сверху,
 * такое что. Эта точка есть сумма векторов current + (left - current) + (top - current)
 *
 *
 * @param current
 * @param prev
 * @param prevVertical
 * @return
 */
Point3f approximate2(Point3f current, Point3f left, Point3f top) {
    auto x = current.x + (left.x - current.x) + (top.x - current.x);
    auto y = current.y + (left.y - current.y) + (top.y - current.y);

    return {x, y, (current.z + left.z + top.z) / 3};
}

float findGrid(
        const Size &size,
        const CenterQuad &quad,
        std::vector<cv::Point3f> &points,
        std::vector<Point3f> &grid,
        size_t *gridWidth,
        size_t *gridHeight
) {
    auto pointsCount = points.size();
    auto imageArea = size.width * size.height;
    auto pixelsPerPoint = std::sqrt(imageArea / pointsCount);
    auto w = (size_t)(size.width / pixelsPerPoint);
    auto h = (size_t)(size.height / pixelsPerPoint);
    *gridWidth = w;
    *gridHeight = h;

    CV_Assert(w * h <= grid.size());

    for (int i = 0; i < grid.size(); ++i) {
        grid[i] = Point3f(0,0,-1);
    }

    auto cH = (int)(h / 2);
    auto cW = (int)(w / 2);

    grid[cH * w + cW] = quad.topLeft;
    grid[cH * w + cW + 1] = quad.topRight;
    grid[(cH + 1) * w + cW] = quad.bottomLeft;
    grid[(cH + 1) * w + cW + 1] = quad.bottomRight;

    // заполним 2 строки сетки влево и вправо от центрального квадрата
    for (auto j = 0; j < 2; j++) {
        fillGridRow(w, cH, cW, j, points, grid);
    }

    auto err = 0.0f;

    // заполним 2 столбца сетки вверх и вниз от центрального квадрата
    for (auto i = 0; i < cH; i++) {
        for (auto s = -1; s <= 1; s += 2) {
            for (auto j = 0; j < 2; j++) {
                auto current = grid[(cH + s * i) * w + cW + j];
                auto prev = grid[(cH + s * (i - 1)) * w + cW + j];
                auto next = &grid[(cH + s * (i + 1)) * w + cW + j];
                auto searchRadius = distance(current, prev) / 4;
                auto nextApproximated = approximate(current, prev);
                *next = findNearestPoint(nextApproximated, points, searchRadius);
                err += (float)std::pow(distance(*next, nextApproximated), 2);
            }
            for (auto k = 0; k < cW; k++) {
                for (auto ks = -1; ks <= 1; ks += 2) {
                    auto current = grid[(cH + s * i) * w + cW + ks * k];
                    auto left = grid[(cH + s * (i + 1)) * w + cW + ks * k];
                    auto top = grid[(cH + s * i) * w + cW + ks * (k + 1)];
                    auto next = &grid[(cH + s * (i + 1)) * w + cW + ks * (k + 1)];
                    auto searchRadius = distance(current, left) / 4;
                    auto nextApproximated = approximate2(current, left, top);
                    *next = findNearestPoint(nextApproximated, points, searchRadius);
                    err += (float)std::pow(distance(*next, nextApproximated), 2);
                }
            }
        }
    }

    cropGrid(grid, gridWidth, gridHeight);

    return std::sqrt(err / (float)(w * h));
}

void cropGrid(std::vector<Point3f> &grid, size_t *w, size_t *h) {
    int top = 0, left = 0, right = 0, bottom = 0;

    for (int y = 0; y < *h; ++y) {
        top = y;
        int count = 0;
        for (int x = 0; x < *w; ++x) {
            count += grid[y * *w + x].z > 0;
        }

        if (count > 2) {
            break;
        }

    }

    for (int y = (int)*h; y > 0; --y) {
        bottom = y;
        int count = 0;
        for (int x = 0; x < *w; ++x) {
            count += grid[(y - 1) * *w + x].z > 0;
        }

        if (count > 2) {
            break;
        }

    }

    for (int x = 0; x < *w; ++x) {
        left = x;
        int count = 0;
        for (int y = 0; y < *h; ++y) {
            count += grid[y * *w + x].z > 0;
        }

        if (count > 2) {
            break;
        }

    }

    for (int x = (int)*w; x > 0; --x) {
        right = x;
        int count = 0;
        for (int y = 0; y < *h; ++y) {
            count += grid[y * *w + x - 1].z > 0;
        }

        if (count > 2) {
            break;
        }

    }

    auto w0 = *w;
    *w = right - left;
    *h = bottom - top;

    for (int y = 0; y < *h; ++y) {
        for (int x = 0; x < *w; ++x) {
            grid[y * *w + x] = grid[(y + top) * w0 + x + left];
        }
    }
}

void fillGridRow(size_t w, int cH, int cW, int j, std::vector<cv::Point3f> &points, std::vector<Point3f> &grid) {
    for (auto i = 0; i < cW; i++) {
        for (auto s = -1; s <= 1; s += 2) {
            auto current = grid[(cH + j) * w + cW + s * i];
            auto prev = grid[(cH + j) * w + cW + s * (i - 1)];
            auto next = &grid[(cH + j) * w + cW + s * (i + 1)];
            auto searchRadius = distance(current, prev) / 3;
            auto nextApproximated = approximate(current, prev);
            *next = findNearestPoint(nextApproximated, points, searchRadius);
        }
    }
}

Point3f findNearestPoint(const Point3f &point, std::vector<cv::Point3f> &points, float searchRadius) {
    auto found = Point3f(0, 0, -1);
    auto foundDistance = 1e6;
    for (auto p : points) {
        if (std::abs(p.x - point.x) > searchRadius || std::abs(p.y - point.y) > searchRadius || p.z < point.z * 0.6) {
            continue;
        }

        auto d = distance(p, point);

        if (d < foundDistance) {
            foundDistance = d;
            found = p;
        }
    }

    if (found.z < 0) {
        return Point3f(point.x, point.y, -1);
    }

    return found;
}

float findQuadBy3Points(const std::vector<cv::Point3f> &points, size_t size, CenterQuad &result) {
    auto q = 1.0f / 0.0f;
    for (int i = 0; i < size; ++i) {
        auto p = points[i];
        if (p.x > result.topLeft.x && p.y > result.topLeft.y) {
            CenterQuad quad;
            quad.topLeft = result.topLeft;
            quad.topRight = result.topRight;
            quad.bottomLeft = result.bottomLeft;
            quad.bottomRight = p;
            auto q2 = quadQualityRms(quad) / p.z;

            if (q2 < q) {
                q = q2;
                result = quad;
            }
        }
    }

    return q;
}

float findQuadByTop(const std::vector<cv::Point3f> &points, size_t size, CenterQuad &result) {
    auto q = 1.0f / 0.0f;
    for (int i = 0; i < size; ++i) {
        auto p = points[i];
        if (p.y > result.topLeft.y && p.y > result.topRight.y) {
            CenterQuad quad;
            quad.topLeft = result.topLeft;
            quad.topRight = result.topRight;
            quad.bottomLeft = p;
            auto q2 = findQuadBy3Points(points, size, quad) / p.z;

            if (q2 < q) {
                q = q2;
                result = quad;
            }
        }
    }

    return q;
}

float findQuadByTopLeft(const std::vector<cv::Point3f> &points, size_t size, CenterQuad &result) {
    auto q = 1.0f / 0.0f;
    for (int i = 0; i < size; ++i) {
        auto p = points[i];
        if (p.x > result.topLeft.x) {
            CenterQuad quad;
            quad.topLeft = result.topLeft;
            quad.topRight = p;
            auto q2 = std::sqrt(distance(p, result.topLeft)) * findQuadByTop(points, size, quad) / p.z;

            if (q2 < q) {
                q = q2;
                result = quad;
            }
        }
    }

    return q;
}

/**
 * Алгоритм:
 * 1. Выбрать область в центре, содержащую 10-15 точек. Предполагается что на 4 истинных точки приходится не более 6 шумовых
 * 2. Перебрать все комбинации, (худший случай не более 15^4, ожидаемо в среднем не более 3000)
 *    и найти ту, которая дает наиболее правильный квадрат, наиболее близко к центру.
 *
 * Допущения принятые при разработке:
 * - точки расположены преимущественно квадратно-гнездовым
 * - все клетки шахматной доски в центре поля распознаны без пропусков (нет ложноотрицательных срабатываний)
 * - среди точек присутствуют лишние, обусловленные шумом точки, их количество в худшем случае не более чем в полтора раза превышает истинные
 * - угол наклона доски вокруг любой оси не более 30 градусов в любую сторону.
 * - Весовой коэффициент (z-координата) истинной точки, как правило, выше веса ложной.
 *
 * Что остается неизвестным:
 * - размер клеток
 * - цвет клеток
 * - наклоны доски по трем осям
 * - четкий порог веса между истинными и ложными (шумовыми) точками. Также не известно можно ли достоверно разграничить таким порогом.
 *
 * @param size размер фрейма
 * @param points список найденных точек, преимущественно расположенных квадратно-гнездовым
 * @param result возвращаемое значение, 4 точки угла опорного квадрата
 * @return нормированная оценка качества полученного квадрата. Квадрат считается идеальным, если его стороны и диагонали / sqrt(2) равны.
 */
float findCenterQuad(Size size, const std::vector<cv::Point3f> &points, CenterQuad &result) {
    auto center = Point3f((float)size.width / 2, (float)size.height / 2, 0);

    result = {Point3f(0, 0, -1), Point3f(0, 0, -1), Point3f(0, 0, -1), Point3f(0, 0, -1)};

    std::vector<cv::Point3f> centralPoints = points;

    auto roiSize = (float)std::min(size.width, size.height) / 4;

    struct Rect {
        float x;
        float y;
        float x2;
        float y2;
    };

    size_t nextSize, currentSize;
    currentSize = points.size();

    do {
        auto roi = Rect{center.x - roiSize, center.y - roiSize, center.x + roiSize, center.y + roiSize};
        nextSize = 0;
        for (auto i = 0; i < currentSize; i++) {
            auto p = centralPoints[i];
            if (p.x > roi.x && p.x < roi.x2 && p.y > roi.y && p.y < roi.y2) {
                centralPoints[nextSize++] = p;
            }
        }
        roiSize *= std::sqrt(10 / (float)nextSize);
        currentSize = nextSize;
    } while (nextSize >= 15);

    auto q = 1.0f / 0.0f;

    for (auto i = 0; i < currentSize; i++) {
        auto p = centralPoints[i];
        if (p.x <= center.x && p.y <= center.y) {
            CenterQuad quad;
            quad.topLeft = p;
            auto q2 = std::sqrt(distance(p, center)) * findQuadByTopLeft(centralPoints, currentSize, quad) / p.z / p.z;

            if (q2 < q) {
                q = q2;
                result = quad;
            }
        }
    }

    return quadQualityNormRms(result);
}

float quadQualityNormRms(CenterQuad quad) {
    auto left = distance(quad.topLeft, quad.bottomLeft);
    auto right = distance(quad.topRight, quad.bottomRight);
    auto top = distance(quad.topLeft, quad.topRight);
    auto bottom = distance(quad.bottomLeft, quad.bottomRight);
    auto d1 = (float)(distance(quad.topLeft, quad.bottomRight) / sqrt(2));
    auto d2 = (float)(distance(quad.bottomLeft, quad.topRight) / sqrt(2));

    float d = (float)(
            std::abs(left - top) / std::sqrt(left * top)
               + std::abs(left - right) / std::sqrt(left * right)
               + std::abs(right - bottom) / std::sqrt(right * bottom)
               + std::abs(top - bottom) / std::sqrt(top * bottom)
               + std::abs(d1 - d2) / std::sqrt(d1 * d2)
           ) / 5;

    return d;
}

float quadQualityRms(CenterQuad quad) {
    auto d1 = (float)(distance(quad.topLeft, quad.bottomRight) / sqrt(2));
    auto d2 = (float)(distance(quad.bottomLeft, quad.topRight) / sqrt(2));

    return quadQualityNormRms(quad) * (d1 + d2) / 2;
}

void createCheckboadPatterns(Mat &t1) {
    CV_Assert(t1.cols == t1.rows);
    t1.setTo(Scalar(0));

    auto size = t1.cols / 2;

    auto t1tl = Mat(t1, Rect(0, 0, size, size));
    auto t1br = Mat(t1, Rect(size, size, size, size));
    t1tl.setTo(Scalar(255));
    t1br.setTo(Scalar(255));
}

/**
 * kernel size = 3
 *   1 1 1
 *   1 2 1
 *   1 1 1
 * kernel size = 5
 *   0 0 0 0 0
 *   0 1 1 1 0
 *   0 1 2 1 0
 *   0 1 1 1 0
 *   0 0 0 0 0
 */

void findPeaks(const Mat &mat, std::vector<cv::Point3f> &points, size_t *size, int kernel, int noiseTolerance) {
    auto data = (float *)mat.data;
    auto p = 0;

    Mat mask = Mat::zeros(mat.size(), CV_8SC1);

    int kernelRadius = (kernel - 1) / 2;
    int w = mat.cols;
    for (int i = kernelRadius; i < mat.rows - kernel; ++i) {
        for (int j = kernelRadius; j < w - kernel; ++j) {
            if (mask.at<char>(i, j) == 1) {
                continue;
            }
            auto found = true;
            for (int k = 1; k <= kernelRadius; ++k) {
                auto diagonals = (data[(i + (k - 1)) * w + (j + (k - 1))] < data[(i + k) * w + (j + k)])
                        + (data[(i + (k - 1)) * w + (j - (k - 1))] < data[(i + k) * w + (j - k)])
                        + (data[(i - (k - 1)) * w + (j + (k - 1))] < data[(i - k) * w + (j + k)])
                        + (data[(i - (k - 1)) * w + (j - (k - 1))] < data[(i - k) * w + (j - k)]);

                auto sideWidth = (k - 1) * 2;

                auto top = 0;
                for (int s = 0; s < sideWidth; ++s) {
                    top += data[(i - (k - 1)) * w + j - (k - 1) + s] < data[(i - k) * w + j - (k - 1) + s];
                }

                auto bottom = 0;
                for (int s = 0; s < sideWidth; ++s) {
                    bottom += data[(i + (k - 1)) * w + j - (k - 1) + s] < data[(i + k) * w + j - (k - 1) + s];
                }

                auto left = 0;
                for (int s = 0; s < sideWidth; ++s) {
                    left += data[(i - (k - 1) + s) * w + j - (k - 1)] < data[(i - (k - 1) + s) * w + j - k];
                }

                auto right = 0;
                for (int s = 0; s < sideWidth; ++s) {
                    right += data[(i - (k - 1) + s) * w + j + (k - 1)] < data[(i - (k - 1) + s) * w + j + k];
                }

                auto r = diagonals + top + bottom + left + right;

                if (r > noiseTolerance) {
                    found = false;
                    break;
                }
            }

            if (found) {
                const Rect2i &roi = cv::Rect(j - kernelRadius, i - kernelRadius, kernel, kernel);
                auto processed = Mat(mask, roi);
                processed.setTo(1);
                auto point = findMassCenter(mat, j, i, 48);
                j+=kernel;
                points[p] = point;
                p++;
                if (p > *size) {
                    return;
                }
            }
        }
    }

    *size = p;
}

cv::Point3f findMassCenter(const Mat &mat, int x, int y, int searchRadius) {
    auto data = (float *)mat.data;
    auto w = mat.cols;
    auto mass = 0.0f;
    auto sumX = 0.0f;
    auto sumY = 0.0f;
    auto count = 0;

    for (int i = -searchRadius; i <= searchRadius; ++i) {
        for (int j = -searchRadius; j <= searchRadius; ++j) {
            int offset = (y + i) * w + x + j;

            if (offset < 0) {
                continue;
            }

            if ((uchar *)(data + offset) >= mat.dataend) {
                continue;
            }

            auto pixelValue = data[offset];
            if (pixelValue <= 0) {
                continue;
            }
             mass += pixelValue;
            sumX += pixelValue * (float)j;
            sumY += pixelValue * (float)i;
            count++;
        }
    }

    auto cX = (float)x + (sumX / mass);
    auto cY = (float)y + (sumY / mass);

    auto height = 0.0f;
    if (mass > 0) {
        height = data[(int)cY * w + (int)cX];
    }

    return {cX, cY, height};
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
