#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif
#include "opencv2/videoio.hpp"
#include "ImageProcessor.h"
#include "CvCommandHandler.h"
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

#include "CalibrateMapper.h"
#include "Calibrator.h"
#include "DisparityEvaluator.h"

using namespace mini_server;
using namespace cv;

static std::atomic running = true;

static BroadcastingServer broadcastingServer;
static CommandServer commandServer;
int findNearestPoint(const Point2f &point, const std::vector<Point3d> &points, double searchRadius);

void invMap(const cv::Mat &src, cv::Mat &dest) {
    if (dest.empty()) {
        dest = Mat(src.size(), src.type());
    }

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            const auto &p = src.at<cv::Point2f>(y, x);
            int row = (int) std::round(p.y);
            int col = (int) std::round(p.x);
            auto &p2 = dest.at<cv::Point2f>(std::max(0, std::min(row, dest.rows - 1)), std::max(0, std::min(col, dest.cols - 1)));
            p2.x = (float)x;
            p2.y = (float)y;
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
    double avgTime2 = 0.;

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
        throw std::runtime_error("OpenCL is not available");
    }

    std::atomic calibrating = false;
    std::atomic<double> calibrationRMS = std::numeric_limits<double>::max();
    std::thread calibration;

    UMat calibLeft, calibRight;

    UMat lFrame, rFrame;
    UMat lDispMap, rDispMap, leftDispMap, rightDispMap;

    std::vector<cv::UMat> frames(2);
    cv::Mat aligned;
    cv::Mat alignedMap;
    cv::Mat invAlignedMap;
    std::vector<cv::Mat> bestMap1(frames.size()), bestMap2(frames.size());
    std::vector<cv::Mat> invBestMap1(frames.size()), invBestMap2(frames.size());

    matread("map0.bin", bestMap1[0]);
    matread("map1.bin", bestMap1[1]);
    matread("mapa.bin", alignedMap);

    std::vector<double> avgDistCoeff(14);
    std::vector<double> prevStdRmseUndistorted(frames.size(), 1e6);
    std::vector<size_t> nextFrameId(frames.size());

    std::vector<ecv::CalibrateMapper<double>> calibrateMapper(frames.size());
    std::vector<ecv::CalibrateMapper<double>> calibrateMapper2(frames.size());
    std::vector<ecv::CalibrateMapper<double>> calibrateMapper3(frames.size());
    std::vector<ecv::Calibrator> calibrator(frames.size());

    std::vector objectPoints(frames.size(), std::vector<std::vector<Point3f>>(3));
    std::vector imagePoints(frames.size(), std::vector<std::vector<Point2f>>(3));
#ifdef HAVE_OPENCV_HIGHGUI
    cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("src " + std::to_string(0), cv::WINDOW_AUTOSIZE);

    auto mouseDisp = cv::Point2i();
    auto mouseSrc = cv::Point2i();

    void (*onMouse)(int, int, int, int, void *) = [](int event, int x, int y, int flags, void *userdata) {
        auto mouse = (cv::Point2i *) userdata;
        mouse->x = x;
        mouse->y = y;
    };
    cv::setMouseCallback("Disparity", onMouse, &mouseDisp);
    cv::setMouseCallback("src " + std::to_string(0), onMouse, &mouseSrc);
#endif
    ecv::DisparityEvaluator disparityEvaluator;

    cv::Mat disparity8;

    for (int i = 0; running; i++) {
        long nextFrame = std::max(readerLeftCount, readerRightCount);
        if (readerLeftCount == frameCount || readerRightCount == frameCount || readingImLeft.empty() || readingImRight.empty()) {
            usleep(1);
            continue;
        }

        frameCount = nextFrame;

        auto start = std::chrono::high_resolution_clock::now();
        std::chrono::system_clock::time_point startDisp;
        std::chrono::system_clock::time_point endDisp;

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

            if (std::isnan(gridRmse)) {
                waitKey(1);
                continue;
            }

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
                    prevStdRmseUndistorted[j] *= 1.01;
                }
                std::cout << "; plainGridRmse " << plainGridRmse << "; stdGridRmse " << plainStdGridRmse << "; prevStdRmseUndistorted " << prevStdRmseUndistorted[j] << std::endl;
            }
#ifdef HAVE_OPENCV_HIGHGUI
            cv::imshow("GL " + std::to_string(j), colorFrame);
#endif
            if (!bestMap1[j].empty()) {
                cv::remap(frame, plain[j], bestMap1[j], bestMap2[j], cv::INTER_NEAREST);
#ifdef HAVE_OPENCV_HIGHGUI
                cv::imshow("Plain best " + std::to_string(j), plain[j]);
#endif
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
#ifdef HAVE_OPENCV_HIGHGUI
                cv::imshow("Aligned", aligned);

                cv::Mat disparityFp;
                cv::Mat disparity;
                cv::Mat variance;
                disparity.setTo(0);
                std::vector<cv::UMat> framesToEval({plain[0].getUMat(AccessFlag::ACCESS_READ), aligned.getUMat(AccessFlag::ACCESS_READ)});

                disparityEvaluator.evaluateDisparity(framesToEval, disparity, variance);

                double minVal = 0, maxVal = 0, varianceMinVal = 0, varianceMaxVal = 0;

                disparity.copyTo(disparityFp);
                if (minVal == 0 || maxVal == 0) {
                    cv::minMaxLoc(disparityFp, &minVal, &maxVal);
                    maxVal = 300 * ecv::DisparityEvaluator::DISPARITY_PRECISION;
                    minVal = 0;
                }

                disparityFp -= minVal;
                disparityFp *= 255.0 / (maxVal - minVal);

                disparityFp.convertTo(disparity8, CV_8U);
                cv::applyColorMap(disparity8, disparity8, ColormapTypes::COLORMAP_JET);

                calibrateMapper[0].drawGrid(disparity8, plainImageGrid[0], gridWidth[0], gridHeight[0], cv::Scalar(255, 255, 255));

                cv::imshow("Disparity", disparity8);
#endif
                for (int j = 0; j < frames.size(); ++j) {
                    invMap(bestMap1[j], invBestMap1[j]);
                }
                invMap(alignedMap, invAlignedMap);

                matwrite("map0.bin", bestMap1[0]);
                matwrite("map1.bin", bestMap1[1]);
                matwrite("mapa.bin", alignedMap);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        double time = ((double) (end - start).count()) / 1e6;
        double time2 = ((double) (endDisp - startDisp).count()) / 1e6;
        double avgA = 2. / ((i < 50 ? 5 : 50) + 1);
        avgTime = avgTime == 0. ? time : avgA * time + (1 - avgA) * avgTime;
        avgTime2 = avgTime2 == 0. ? time2 : avgA * time2 + (1 - avgA) * avgTime2;
        avgFps = avgFps == 0. ? fps : avgA * fps + (1 - avgA) * avgFps;

        std::cerr << "fps " << avgFps << " time " << time << " time2 " << time2 << " load " << avgTime * 100 / ((double)us / 1e6) << " %, avg " << avgTime << " %, avg2 " << avgTime2 << " size "
                  << resultLeft.dataend - resultLeft.datastart << std::endl;

        std::ostringstream tm;

        tm << "PERF " << fps << " " << time << std::endl;

#ifdef HAVE_OPENCV_HIGHGUI

        if (cv::waitKey(1) >= 0) {
            break;
        }
#endif
    }

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
