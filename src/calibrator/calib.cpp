#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
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
#include "CalibrateFrameCollector.h"
#include "CalibrationStrategy.h"
#include <common/RemoteView.h>
#include <common/Telemetry.h>

int main(int argc, const char **argv) {
    cpptrace::register_terminate_handler();

    cv::Mat captureFrameLeft, captureFrameRight, readingLeft, readingRight, readingImLeft, readingImRight;
    cv::UMat output, outputLeft, inputLeft, imageLeft, outputRight, inputRight, imageRight;
    cv::Mat resultLeft, resultRight;
    cv::VideoCapture captureLeft, captureRight;

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
        captureLeft.open(std::string(argv[1]), cv::CAP_V4L2, params);
        captureRight = captureLeft;
    } else {
        captureLeft.open(std::string(argv[1]), cv::CAP_V4L2, params);
        captureRight.open(std::string(argv[2]), cv::CAP_V4L2, params);
    }

    auto remoteView = ecv::RemoteView();

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

    auto tm = std::shared_ptr<mini_server::BroadcastingServer>(new mini_server::BroadcastingServer);
    tm->setSocket(mini_server::SocketFactory::createListeningSocket("/tmp/cv_tm", 10));
    std::thread tmThread([&tm] () {
        tm->run();
    });

    ecv::Telemetry::setServer(tm);
    ecv::Telemetry::setLogLevel(ecv::Telemetry::LogLevel::DEBUG);

    if (!captureLeft.isOpened()) {
        std::cerr << "Left Capture is not opened" << std::endl;
        return -1;
    }

    if (!captureRight.isOpened()) {
        std::cerr << "Right Capture is not opened" << std::endl;
        return -1;
    }

    if (!cv::ocl::useOpenCL()) {
        throw std::runtime_error("OCL not available");
    }

    std::vector<cv::VideoCapture> captures({captureLeft, captureRight});
    std::vector<cv::UMat> frames(captures.size());
    std::vector<cv::Mat> plainFrames(frames.size());
    std::vector<cv::Mat> rectifiedFrames(frames.size());
    std::vector<std::mutex> framesMutex(frames.size());
    std::vector<cv::Mat> readingFrames(frames.size());
    std::vector<cv::Mat> readFrames(frames.size());
    std::vector<double> frameTs(frames.size());
    std::vector<double> getFrameTs(frames.size());
    std::vector<std::thread> readerThreads(frames.size());
    std::atomic<bool> captureRunning = true;

    for (int i = 0; i < frames.size(); ++i) {
        captures[i].read(frames[i]);
    }

    auto captureThreadCallback = [&captures, &framesMutex, &readingFrames, &frameTs, &readFrames, &captureRunning](int i) {
        while (captureRunning) {
            captures[i].read(readingFrames[i]);
            std::lock_guard lock(framesMutex[i]);
            readingFrames[i].copyTo(readFrames[i]);
            frameTs[i] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        }
    };

    for (int i = 0; i < readerThreads.size(); ++i) {
        getFrameTs[i] = 0;
        readerThreads[i] = std::thread(captureThreadCallback, i);
    }

    const cv::Size &size = frames[0].size();
    std::vector<ecv::CalibrateMapper> calibrateMapper(frames.size()), testCalibrateMapper(frames.size());
    std::vector<ecv::Calibrator> calibrator(frames.size());
    std::vector<ecv::CalibrationData> calibrationData(frames.size(), ecv::CalibrationData(size));
    std::vector<ecv::CalibrationData> rectificationData(frames.size(), ecv::CalibrationData(size));

    std::vector<double> bestQ(frames.size(), 1. / 0.);
    std::vector<double> bestPairQ(frames.size(), 1. / 0.);
    std::vector<double> bestRoiQ(frames.size(), 0.);
    std::vector<size_t> bestW(frames.size(), 0);
    std::vector<size_t> bestH(frames.size(), 0);
    std::unordered_map<int, std::vector<ecv::CalibrateMapper::Point3>> framesMap;

    std::vector<double> progress(frames.size(), 0);

    std::mutex mapsMutex;
    std::vector<cv::Mat> maps(frames.size());
    std::vector<cv::Mat> rectifiedMaps(frames.size());

    ecv::CalibrationStrategy calibrationStrategy(size, (int)frames.size(), [&maps, &rectifiedMaps, &mapsMutex, &calibrationData, &rectificationData](int cameraId, const ecv::CalibrationStrategy &that) {
        {
            std::unique_lock lock(mapsMutex);
            maps[cameraId] = that.getMap(cameraId).clone();
            rectifiedMaps[cameraId] = that.getRectifiedMap(cameraId).clone();
            calibrationData[cameraId] = that.getCalibrationData(cameraId);
            rectificationData[cameraId] = that.getRectificationData(cameraId);
        }

        std::cout << "Calibration updated " << cameraId << ",\t" << that.getProgress(cameraId) << "%\tcost " << that.getViewCosts(cameraId) << std::endl;
    });

//    calibrationStrategy.loadConfig();
    calibrationStrategy.runCalibration();

    while (true) {
        bool hasNewFrames = false;
        for (int i = 0; i < frames.size(); ++i) {
            if (frameTs[i] > getFrameTs[i]) {
                framesMutex[i].lock();
                if (i == 1) {
                    cv::rotate(readFrames[i], frames[i], cv::RotateFlags::ROTATE_180);
                } else {
                    readFrames[i].copyTo(frames[i]);
                }
                getFrameTs[i] = frameTs[i];
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

        std::vector<std::vector<ecv::CalibrateMapper::Point3>> peaks(frames.size(), std::vector<ecv::CalibrateMapper::Point3>(500));
#pragma omp parallel for default(none) shared(peaks, frames, calibrateMapper)
        for (int i = 0; i < frames.size(); ++i) {
            size_t peaksSize;

            calibrateMapper[i].detectPeaks(frames[i], peaks[i], &peaksSize);
            peaks[i].resize(peaksSize);
        }

        std::vector<ecv::CalibrateFrameCollector::FrameRef> frameSet(frames.size(), nullptr);

        for (int i = 0; i < frames.size(); ++i) {
            cv::Mat debug;
            frames[i].copyTo(debug);
            std::vector<ecv::CalibrateMapper::Point3> imageGrid(500), objectGrid(500);
            std::vector<ecv::CalibrateMapper::Point3> plainGrid(500);
            std::vector<ecv::CalibrateMapper::Point3> rectifiedGrid(500);
            int w = 0, h = 0;

            auto frameQuality = calibrateMapper[i].detectFrameImagePointsGrid(frames[i], peaks[i], imageGrid, &w, &h, debug);

            std::vector<ecv::CalibrateMapper::Point3> calibImageGrid(500), calibObjectGrid(500);

            if (i == 1) {
                ecv::Telemetry::status("calibration", "grid.error.raw", std::format("Grid error is {}", frameQuality));
            }

            if (frameQuality < 1 && w >= 6 && h >= 3) {
                calibrateMapper[i].generateFrameObjectPointsGrid(objectGrid, w, h);
                imageGrid.resize(w * h);
                objectGrid.resize(w * h);

                frameSet[i] = calibrationStrategy.createFrame(i, imageGrid, objectGrid, (int)w, (int)h, frameQuality, getFrameTs[i]);
            }

            cv::Mat map;
            cv::Mat rectifiedMap;
            ecv::CalibrationData calibrationDatum, rectificationDatum;
            {
                std::unique_lock lock(mapsMutex);
                map = maps[i].clone();
                rectifiedMap = rectifiedMaps[i].clone();
                calibrationDatum = calibrationData[i];
                rectificationDatum = rectificationData[i];
            }

            if (!maps[i].empty()) {
                cv::remap(frames[i], plainFrames[i], map, cv::noArray(), cv::INTER_NEAREST);
            } else {
                frames[i].copyTo(plainFrames[i]);
            }
            if (!rectifiedMaps[i].empty()) {
                cv::remap(frames[i], rectifiedFrames[i], rectifiedMap, cv::noArray(), cv::INTER_NEAREST);
            } else {
                frames[i].copyTo(rectifiedFrames[i]);
            }

            calibrationStrategy.undistortImagePoints(imageGrid, plainGrid, calibrationDatum);
            calibrationStrategy.rectifyImagePoints(imageGrid, rectifiedGrid, rectificationDatum);

            calibrateMapper[i].drawGrid(plainFrames[i], plainGrid, w, h, cv::Scalar(255, 0, 0));
            calibrateMapper[i].drawGrid(rectifiedFrames[i], rectifiedGrid, w, h, cv::Scalar(255, 0, 0));

            const cv::String &text = std::format(
                    "sz = {}x{} ({}x{})\nprogress = {}% (f={}, s={})\nq = {}\nbestQ = {}\nmcBestQ = {}\npatternSize = {}\npatternSkew = {}\nf = {}x{}\nc = {}x{}",
                    w, h, calibrationStrategy.getGridSize().first,
                    calibrationStrategy.getGridSize().second,
                    calibrationStrategy.getProgress(i) * 100,
                    calibrationStrategy.getFrameCount(i),
                    calibrationStrategy.getFrameSetCount(0),
                    frameQuality,
                    calibrationStrategy.getViewCosts(i),
                    calibrationStrategy.getViewMulticamCosts(i),
                    calibrateMapper[i].patternSize,
                    calibrateMapper[i].skew,
                    std::round(calibrationStrategy.getF(i).x * 1000) / 1000,
                    std::round(calibrationStrategy.getF(i).y * 1000) / 1000,
                    std::round(calibrationStrategy.getC(i).x * 1000) / 1000,
                    std::round(calibrationStrategy.getC(i).y * 1000) / 1000
            );

            cv::putText(debug, text, cv::Point2i(30, 30), 1, 2, cv::Scalar(0, 0, 255));

#ifdef HAVE_OPENCV_HIGHGUI
            if (!plainFrames[i].empty()) {
                int height = plainFrames[i].rows;
                int step = height / (10 + 1);
                for (int j = 1; j <= 10; ++j) {
                    int y = j * step;
                    cv::line(rectifiedFrames[i], cv::Point(0, y), cv::Point(rectifiedFrames[i].cols - 1, y), cv::Scalar(0, 255, 0), 1);
                }

                remoteView.showMat(std::format("Plain {}", i), plainFrames[i]);
            }
            if (!rectifiedFrames[i].empty()) {
                remoteView.showMat(std::format("Rectified {}", i), rectifiedFrames[i]);
            }

            if (!debug.empty()) {
                remoteView.showMat(std::format("Debug {}", i), debug);
            }
#endif
        }

        calibrationStrategy.addFrameSet(frameSet);

#ifdef HAVE_OPENCV_HIGHGUI
        if (remoteView.waitKey() != -1) {
            break;
        }
#endif
    }

    captureRunning = false;

    for (auto &thread : readerThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    calibrationStrategy.stopCalibration();

    return 0;
}