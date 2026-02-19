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
#include "../../MatStorage.h"
#include "CalibrateFrameCollector.h"
#include "BlurFrameFilter.h"
#include "CalibrationStrategy.h"

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
        captureLeft.open(std::string(argv[1]), cv::CAP_V4L2, params);
        captureRight = captureLeft;
    } else {
        captureLeft.open(std::string(argv[1]), cv::CAP_V4L2, params);
        captureRight.open(std::string(argv[2]), cv::CAP_V4L2, params);
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

    if (!cv::ocl::useOpenCL()) {
        throw std::runtime_error("OCL not available");
    }

    std::vector<cv::VideoCapture> captures({captureLeft, captureRight});
    std::vector<cv::UMat> frames(captures.size());
    std::vector<cv::Mat> plainFrames(frames.size());
    std::vector<std::mutex> framesMutex(frames.size());
    std::vector<cv::Mat> readingFrames(frames.size());
    std::vector<cv::Mat> readFrames(frames.size());
    std::vector<double> frameTs(frames.size());
    std::vector<double> getFrameTs(frames.size());
    std::vector<std::thread> readerThreads(frames.size());
    volatile bool threadRunning[frames.size()];


//    ecv::DisparityEvaluator disparityEvaluator;
//    disparityEvaluator.lazyInitializeOcl();
//    cv::UMat left, right;
//
//    disparityEvaluator.evaluateDisparity(left, right);


    auto captureThreadCallback = [&captures, &framesMutex, &readingFrames, &frameTs, &readFrames, &threadRunning](int i) {
        while (threadRunning[i]) {
            captures[i].read(readingFrames[i]);
            framesMutex[i].lock();
            readingFrames[i].copyTo(readFrames[i]);
            frameTs[i] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            framesMutex[i].unlock();
        }
    };

    for (int i = 0; i < readerThreads.size(); ++i) {
        threadRunning[i] = true;
        getFrameTs[i] = 0;
        readerThreads[i] = std::thread(captureThreadCallback, i);
    }

    std::vector<ecv::CalibrateMapper> calibrateMapper(frames.size()), testCalibrateMapper(frames.size());
    std::vector<ecv::BlurFrameFilter> blurFrameFilter(frames.size(), ecv::BlurFrameFilter(0.2));
    std::vector<ecv::Calibrator> calibrator(frames.size());
    std::vector<ecv::Calibrator::CalibrationData> calibrationData(frames.size());

    std::mutex mapsMutex;
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

    std::vector<double> frameBlur(frames.size());
    std::vector<double> bestQ(frames.size(), 1. / 0.);
    std::vector<double> bestPairQ(frames.size(), 1. / 0.);
    std::vector<double> bestRoiQ(frames.size(), 0.);
    std::vector<size_t> bestW(frames.size(), 0);
    std::vector<size_t> bestH(frames.size(), 0);
    std::unordered_map<int, std::vector<ecv::CalibrateMapper::Point3>> framesMap;

    std::vector<double> progress(frames.size(), 0);

    std::shared_ptr<ecv::CalibrateFrameCollector::Frame> baseFrameRef;

    const cv::Size &size = frames[0].size();
    ecv::CalibrationStrategy calibrationStrategy(size, (int)frames.size(), [&maps, &mapsMutex, &progress](int cameraId, const ecv::CalibrationStrategy &that) {
        {
            std::unique_lock lock(mapsMutex);
            maps[cameraId] = that.getMap(cameraId);
        }
        progress[cameraId] = that.getProgress(cameraId);

        std::cout << "Calibration updated " << cameraId << ",\t" << that.getProgress(cameraId) << "%\tcost " << that.getCosts(cameraId) << std::endl;
    });

    calibrationStrategy.loadConfig();
    calibrationStrategy.runCalibration();

    while (true) {
        bool hasNewFrames = false;
        for (int i = 0; i < frames.size(); ++i) {
            if (frameTs[i] > getFrameTs[i]) {
                framesMutex[i].lock();
                readFrames[i].copyTo(frames[i]);
                getFrameTs[i] = frameTs[i];
                framesMutex[i].unlock();
                hasNewFrames = true;
                if (i == 1) {
                    cv::rotate(frames[i], frames[i], cv::RotateFlags::ROTATE_180);
                }
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

        baseFrameRef.reset();

        bool abortCalibration[frames.size()];
        for (int i = 0; i < frames.size(); ++i) {
            abortCalibration[i] = false;
        }

        std::vector<std::vector<ecv::CalibrateMapper::Point3>> peaks(frames.size(), std::vector<ecv::CalibrateMapper::Point3>(500));
#pragma omp parallel for default(none) shared(frameBlur, peaks, frames, calibrateMapper, blurFrameFilter, abortCalibration, std::cout)
        for (int i = 0; i < frames.size(); ++i) {
            frameBlur[i] = blurFrameFilter[i].getValue(frames[i]);
//            if (!blurFrameFilter[i].streamingPercentile(frameBlur[i])) {
//                abortCalibration[i] = true;
//                continue;
//            }

            size_t peaksSize;

            calibrateMapper[i].detectPeaks(frames[i], peaks[i], &peaksSize);
            peaks[i].resize(peaksSize);
        }

        std::vector<ecv::CalibrateFrameCollector::FrameRef> frameSet(frames.size(), nullptr);

        for (int i = 0; i < frames.size(); ++i) {
            cv::Mat debug;
            frames[i].copyTo(debug);
            std::vector<ecv::CalibrateMapper::Point3> imageGrid(500), objectGrid(500);
            size_t w = 0, h = 0;

            if (!maps[i].empty()) {
                cv::Mat map;
                {
                    std::unique_lock lock(mapsMutex);
                    map = maps[i];
                }
                cv::remap(frames[i], plainFrames[i], map, cv::noArray(), cv::INTER_NEAREST);
            } else {
                frames[i].copyTo(plainFrames[i]);
            }

            if (!abortCalibration[i]) {
                auto srcGridQ = calibrateMapper[i].detectFrameImagePointsGrid(frames[i], peaks[i], imageGrid, &w, &h, debug);

                if (w > 0 && h > 0) {
                    calibrateMapper[i].generateFrameObjectPointsGrid(objectGrid, w, h);
                    calibrateMapper[i].drawGrid(debug, objectGrid, w, h, cv::Scalar(255, 0, 255), 2);
                }

                double calibGridQ = 1. / 0.;
                std::vector<ecv::CalibrateMapper::Point3> calibImageGrid(500), calibObjectGrid(500);
                cv::Mat map;
                cv::Mat testFrame;

                if (srcGridQ < 0.03 && w >= 6 && h >= 4) {
                    imageGrid.resize(w * h);
                    objectGrid.resize(w * h);

                    frameSet[i] = calibrationStrategy.createFrame(i, imageGrid, objectGrid, (int)w, (int)h, frameBlur[i], getFrameTs[i]);
                }

                const cv::String &text = std::format(
                        "sz = {}x{}\nsrcGridQ = {}\nbestQ = {}\nmcBestQ = {}\npatternSize = {}\npatternSkew = {}\nprogress = {}%",
                        w, h, srcGridQ, calibrationStrategy.getCosts(i), calibrationStrategy.getMulticamCosts(),
                        calibrateMapper[i].patternSize, calibrateMapper[i].skew,
                        calibrationStrategy.getProgress(i) * 100
                );
                cv::putText(debug, text, cv::Point2i(30, 30), 1, 2, cv::Scalar(0, 0, 255));
            }

#ifdef HAVE_OPENCV_HIGHGUI
            if (!plainFrames[i].empty()) {
                int height = plainFrames[i].rows;
                int step = height / (10 + 1);
                for (int j = 1; j <= 10; ++j) {
                    int y = j * step;
                    cv::line(plainFrames[i], cv::Point(0, y), cv::Point(plainFrames[i].cols - 1, y), cv::Scalar(0, 255, 0), 1);
                }

                cv::imshow(std::format("Plain {}", i), plainFrames[i]);
            }
            if (!frames[i].empty()) {
                cv::imshow(std::format("Camera {}", i), frames[i]);
            }

            if (!debug.empty() && !abortCalibration[i]) {
                cv::imshow(std::format("Debug {}", i), debug);
            }
#endif
        }

        calibrationStrategy.addFrameSet(frameSet);

#ifdef HAVE_OPENCV_HIGHGUI
        if (cv::waitKey(1) != -1) {
            break;
        }
#endif
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