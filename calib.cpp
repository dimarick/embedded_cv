#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif
#include <iostream>
#include <chrono>
#include <cpptrace/cpptrace.hpp>
#include <BroadcastingServer.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <SocketFactory.h>
#include <core/ocl.hpp>
#include "Calibrator.h"
#include "CalibrateMapper.h"
#include "MatStorage.h"

void rotate180(cv::Mat &map)
{
    for (int y = 0; y < map.rows; ++y) {
        for (int x = 0; x < map.cols; ++x) {
            auto p = map.at<cv::Point2f>(y, x);
            p.x = (float)map.cols - (float)x;
            p.y = (float)map.rows - (float)y;
            map.at<cv::Point2f>(y, x) = p;
        }
    }
}

void noAction(cv::Mat &map)
{
    for (int y = 0; y < map.rows; ++y) {
        for (int x = 0; x < map.cols; ++x) {
            auto p = map.at<cv::Point2f>(y, x);
            p.x = (float)x;
            p.y = (float)y;
            map.at<cv::Point2f>(y, x) = p;
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

    std::vector<cv::VideoCapture> captures({captureLeft, captureRight});
    std::vector<cv::Mat> frames(captures.size());
    std::vector<cv::Mat> plainFrames(frames.size());
    std::vector<std::mutex> framesMutex(frames.size());
    std::vector<cv::Mat> readingFrames(frames.size());
    std::vector<cv::Mat> readFrames(frames.size());
    std::vector<long> frameIds(frames.size());
    std::vector<long> getFrameIds(frames.size());
    std::vector<std::thread> readerThreads(frames.size());
    volatile bool threadRunning[frames.size()];


//    ecv::DisparityEvaluator disparityEvaluator;
//    disparityEvaluator.lazyInitializeOcl();
//    cv::UMat left, right;
//
//    disparityEvaluator.evaluateDisparity(left, right);


    auto captureThreadCallback = [&captures, &framesMutex, &readingFrames, &frameIds, &readFrames, &threadRunning](int i) {
        while (threadRunning[i]) {
            captures[i].read(readingFrames[i]);
            framesMutex[i].lock();
            readingFrames[i].copyTo(readFrames[i]);
            frameIds[i]++;
            framesMutex[i].unlock();
        }
    };

    for (int i = 0; i < readerThreads.size(); ++i) {
        threadRunning[i] = true;
        frameIds[i] = 0;
        getFrameIds[i] = 0;
        readerThreads[i] = std::thread(captureThreadCallback, i);
    }

    std::vector<ecv::CalibrateMapper<double>> calibrateMapper(frames.size());
    std::vector<ecv::Calibrator> calibrator(frames.size());

    std::vector<cv::Mat> maps(frames.size());

    for (int i = 0; i < frames.size(); ++i) {
        ecv::MatStorage::matRead(std::format("map_{}.bin", i), maps[i]);
    }

    for (int i = 0; i < frames.size(); ++i) {
        captures[i].read(frames[i]);
    }

    for (int i = 0; i < frames.size(); ++i) {
        if (maps[i].empty()) {
            maps[i] = cv::Mat(frames[i].size(), CV_32FC2);
            maps[i].setZero();
            noAction(maps[i]);
        }
    }

    rotate180(maps[1]);


    while (true) {
        bool hasNewFrames = false;
        for (int i = 0; i < frames.size(); ++i) {
            if (frameIds[i] > getFrameIds[i]) {
                framesMutex[i].lock();
                readFrames[i].copyTo(frames[i]);
                getFrameIds[i] = frameIds[i];
                framesMutex[i].unlock();
                hasNewFrames = true;
            }
        }

        bool allFramesValid = true;

        for (const auto & frame : frames) {
            allFramesValid &= !frame.empty();
        }

        if (!hasNewFrames || !allFramesValid) {
            usleep(10000);
            continue;
        }

        double ema = 2. / (10. + 1.);
        double avgRemapTime = 0., avgDetectGridTime = 0.;
        long lastShow = 0;

        for (int i = 0; i < plainFrames.size(); ++i) {
            std::vector<ecv::CalibrateMapper<double>::Point3> imageGrid(500), objectGrid(500);
            size_t w = 0, h = 0;
            double gridQ = 0;

            auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            cv::remap(frames[i], plainFrames[i], maps[i], cv::noArray(), cv::INTER_NEAREST);
            auto remapTime = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            auto srcGridQ = calibrateMapper[i].detectFrameImagePointsGrid(plainFrames[i], imageGrid, &w, &h, plainFrames[i]);

            if (w > 0 && h > 0) {
                gridQ = calibrateMapper[i].generateFrameObjectPointsGrid(imageGrid, objectGrid, w, h);
                calibrateMapper[i].drawGrid(plainFrames[i], objectGrid, w, h, cv::Scalar(255, 0, 255), 2);
            }

            auto detectGridTime = std::chrono::high_resolution_clock::now().time_since_epoch().count();

            avgRemapTime = ema * (double)(remapTime - start) + (1 - ema) * avgRemapTime;
            avgDetectGridTime = ema * (double)(detectGridTime - remapTime) + (1 - ema) * avgDetectGridTime;

            cv::putText(plainFrames[i], std::format("sz = {}x{}\nsrcGridQ = {}\ngridQ = {}\npatternSize = {}\npatternSkew = {}",
                                                    w, h, srcGridQ, gridQ, calibrateMapper[i].patternSize, calibrateMapper[i].skew), cv::Point2i(30, 30), 1, 2,
                        cv::Scalar(0,0,255));

            cv::imshow(std::format("Camera {}", i), plainFrames[i]);
        }

        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        if ((now - lastShow) > (long)1e9) {
            lastShow = now;

            std::cout << std::format("avgRemapTime = {} ms, avgDetectGridTime = {} ms",
                                     (double)avgRemapTime / 1e6, (double)avgDetectGridTime / 1e6) << std::endl;
        }

        if (cv::waitKey(1) != -1) {
            break;
        }
    }

    for (int i = 0; i < readerThreads.size(); ++i) {
        threadRunning[i] = false;
    }

    for (auto &thread : readerThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    return 0;
}