#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif
#include <iostream>
#include <chrono>
#include <cpptrace/cpptrace.hpp>
#include <CommandServer.h>
#include <IpcServer.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <SocketFactory.h>
#include <csignal>
#include <core/opencl/ocl_defs.hpp>
#include <core/ocl.hpp>
#include <common/RemoteView.h>
#include <common/CaptureInfo.h>

#include "DisparityEvaluator.h"
#include "common/Defer.h"
#include "common/MatStorage.h"

using namespace mini_server;

static std::atomic running = true;

static IpcServer tmServer;
static IpcServer streamingServer;
static IpcServer commandServer;
static IpcServer captureServer;

void invMap(const cv::Mat &src, cv::Mat &dest) {
    if (dest.empty()) {
        dest = cv::Mat(src.size(), src.type());
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
    std::cerr << "Built with OpenCV " << CV_VERSION << std::endl;
    auto remoteView = ecv::RemoteView("/tmp/cv_stream");

    if (argc < 2) {
        std::cout << "too less command arguments";

        return -1;
    }

    tmServer.setSocket(SocketFactory::createServerSocket("/tmp/cv_tm", 10));
    streamingServer.setSocket(SocketFactory::createServerSocket("/tmp/cv_stream", 10));
    commandServer.setSocket(SocketFactory::createServerSocket("/tmp/cv_ctl", 1));
    commandServer.setOnMessage([] (int socket, const std::string &message) {});

    signal(SIGINT, [](int signal) {
        running = false;
        captureServer.stop();
        streamingServer.stop();
        tmServer.stop();
        commandServer.stop();
    });

    std::thread commandServerThread = std::thread([]() {
        commandServer.serve();
    });

    std::thread tmServerThread = std::thread([]() {
        tmServer.serve();
    });

    std::thread streamingServerThread = std::thread([]() {
        streamingServer.serve();
    });

    double fps = 0.;
    double avgFps = 0.;
    double avgTime = 0.;
    double avgTime2 = 0.;

    std::mutex readerLock;
    std::atomic nextFrame = 0l;
    long frameCount = 0l;

    std::vector<cv::Mat> frameSet;
    std::vector<cv::Mat> readingFrames;

    auto onFrameSetReceived =
            [&readingFrames, &nextFrame, &readerLock](int socket, const void *buffer, size_t size) {
                auto header = ecv::CaptureBuffer::getHeader(buffer, size);
                if (header == nullptr) {
                    return;
                }
                auto info = header->getFirstCaptureInfo();
                if (info == nullptr) {
                    return;
                }
                auto imageData = info->getImageData();
                std::lock_guard lock(readerLock);

                readingFrames.resize(header->nCaptures);

                for (int i = 0; i < header->nCaptures; i++) {
                    readingFrames[i] = cv::Mat(info->h, info->w, CV_8UC(info->channels), (void *)imageData);
                    info = info->getNextCaptureInfo();
                    imageData = info->getImageData();
                }

                nextFrame++;
            };

    auto captureSocket = mini_server::SocketFactory::createClientSocket(argv[1]);
    captureServer.setSocket(captureSocket);
    captureServer.setOnMessage(onFrameSetReceived);

    auto captureThread = std::thread([&captureServer]() {captureServer.runClient();});

    cv::ocl::setUseOpenCL(true);

#ifndef HAVE_OPENCL
    throw std::runtime_error("OpenCV built without opencl");
#endif

    if (!cv::ocl::haveOpenCL()) {
        throw std::runtime_error("OpenCL is not available");
    }

    cv::UMat calibLeft, calibRight;

    cv::UMat lFrame, rFrame;
    cv::UMat lDispMap, rDispMap, leftDispMap, rightDispMap;

    std::vector<cv::UMat> frames(2);
    std::vector<cv::Mat> invMaps(frames.size()), invBestMap2(frames.size());

    std::vector<cv::UMat> maps;

    for (int i = 0; i < 10; i++) {
        cv::UMat map;
        if (!ecv::MatStorage::matRead(std::format("map{}.bin", i), map)) {
            break;
        }
        maps.emplace_back(map);
    }

    for (int i = 0; i < maps.size(); ++i) {
        invMap(maps[i].getMat(cv::ACCESS_READ), invMaps[i]);
    }

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
    ecv::DisparityEvaluator disparityEvaluator((cl_context)cv::ocl::Context::getDefault().ptr());

    double minVal = 0, maxVal = 0, varianceMinVal = 0, varianceMaxVal = 0;

    cv::Mat disparity8;

    cv::Mat preview;
    std::mutex sendingLock;
    std::ostringstream tm;
    std::vector<cv::UMat> result(frames.size());
    cv::FileStorage fs;
    cv::FileStorage fs_write;
    auto clahe = cv::createCLAHE(1, cv::Size(3,3));

    auto prev = std::chrono::high_resolution_clock::now();

    for (int i = 0; running; i++) {
        if (nextFrame == frameCount || readingFrames.empty()) {
            usleep(1);
            continue;
        }

        frameCount = nextFrame;

        auto start = std::chrono::high_resolution_clock::now();
        std::chrono::system_clock::time_point startDisp;
        std::chrono::system_clock::time_point endDisp;

        readerLock.lock();

        frames.resize(readingFrames.size());
        frameSet.resize(readingFrames.size());

        for (int j = 0; j < readingFrames.size(); j++) {
            readingFrames[j].copyTo(frames[j]);
            readingFrames[j].copyTo(frameSet[j]);
        }

        readerLock.unlock();

        cv::rotate(frames[1], frames[1], cv::ROTATE_180);
        cv::rotate(frameSet[1], frameSet[1], cv::ROTATE_180);

        auto now = std::chrono::high_resolution_clock::now();
        auto us = (double) (now - prev).count();
        prev = now;

        fps = 1e9 / us;

        if (frames[0].empty()) {
            break;
        }

        for (int j = 0; j < frames.size(); ++j) {
#ifdef HAVE_OPENCV_HIGHGUI
            cv::UMat filtered;
            cv::UMat original;
            frames[0].copyTo(original);

            cv::cvtColor(frames[0], original, cv::COLOR_RGB2GRAY);

            clahe->apply(original, filtered);
            clahe->apply(filtered, filtered);

            remoteView.showMat("Original", original);
            remoteView.showMat("Filtered", filtered);

            cv::remap(frames[j], result[j], maps[j], cv::noArray(), cv::INTER_NEAREST);
#endif
        }

#ifdef HAVE_OPENCV_HIGHGUI
        std::vector<cv::Mat> result2(result.size());
        for (auto j = 0; j < result2.size(); j++) {
            result[j].getMat(cv::ACCESS_RW).copyTo(result2[j]);
        }
        cv::drawMarker(frames[0], mouseSrc, cv::Scalar(255, 0, 255), cv::MarkerTypes::MARKER_CROSS, 30, 2);
        auto plainMap1 = invMaps[0].ptr<cv::Point2f>(mouseSrc.y, mouseSrc.x);
        auto plainMap2 = maps[0].getMat(cv::ACCESS_READ).ptr<cv::Point2f>((int)plainMap1->y, (int)plainMap1->x);
        cv::drawMarker(result2[0], cv::Point2i((int) plainMap1->x, (int) plainMap1->y), cv::Scalar(0, 255, 0), cv::MarkerTypes::MARKER_TILTED_CROSS, 30, 2);
        cv::drawMarker(frames[1], cv::Point2i((int) plainMap2->x, (int) plainMap2->y), cv::Scalar(0, 255, 0), cv::MarkerTypes::MARKER_CROSS, 30, 2);
        cv::drawMarker(result2[1], cv::Point2i((int) plainMap1->x, (int) plainMap1->y), cv::Scalar(0, 255, 0), cv::MarkerTypes::MARKER_TILTED_CROSS, 30, 2);

        cv::Mat varianceFp;
        cv::Mat variance8;
#endif
        cv::Mat disparityFp;
        cv::Mat disparity;
        cv::Mat variance;
        disparity.setTo(0);
        variance.setTo(0);

        startDisp = std::chrono::high_resolution_clock::now();
        disparityEvaluator.evaluateDisparity(result, disparity, variance);
        endDisp = std::chrono::high_resolution_clock::now();

        disparity.copyTo(disparityFp);
        if (minVal == 0 || maxVal == 0) {
            cv::minMaxLoc(disparityFp, &minVal, &maxVal);
            maxVal = 300 * ecv::DisparityEvaluator::DISPARITY_PRECISION;
            minVal = 0;
        }

        disparityFp -= minVal;
        disparityFp *= 255.0 / (maxVal - minVal);


        disparityFp.convertTo(disparity8, CV_8U);
        cv::applyColorMap(disparity8, disparity8, cv::ColormapTypes::COLORMAP_JET);
#ifdef HAVE_OPENCV_HIGHGUI
        variance.copyTo(varianceFp);
        variance.copyTo(varianceFp);

        cv::minMaxLoc(variance, &varianceMinVal, &varianceMaxVal);

        varianceFp -= varianceMinVal;
        varianceFp *= 255.0 / (varianceMaxVal - varianceMinVal);

        varianceFp.convertTo(variance8, CV_8U);
        cv::applyColorMap(variance8, variance8, cv::ColormapTypes::COLORMAP_JET);

        //TODO перенести на клиента
        cv::drawMarker(disparity8, mouseDisp, cv::Scalar(255, 128, 255), cv::MarkerTypes::MARKER_CROSS, 30, 3);
        cv::drawMarker(result2[0], mouseDisp, cv::Scalar(255, 128, 255), cv::MarkerTypes::MARKER_CROSS, 30, 3);
        cv::drawMarker(result2[1], mouseDisp, cv::Scalar(255, 128, 255), cv::MarkerTypes::MARKER_CROSS, 30, 3);
        cv::drawMarker(variance8, mouseDisp, cv::Scalar(255, 128, 255), cv::MarkerTypes::MARKER_CROSS, 30, 3);
        auto disparityAtPoint = disparity.at<int16_t>(mouseDisp.y, mouseDisp.x);
        auto varianceAtPoint = variance.at<float>(mouseDisp.y, mouseDisp.x);
        auto dispStr = std::to_string((float)disparityAtPoint / ecv::DisparityEvaluator::DISPARITY_PRECISION);
        auto varStr = std::to_string((float)varianceAtPoint);
        cv::putText(disparity8, dispStr, mouseDisp, cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(255, 192, 255));
        cv::putText(variance8, varStr, mouseDisp, cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(255, 192, 255));
        remoteView.showMat("Disparity", disparity8);
        remoteView.showMat("Variance", variance8);
        for (int j = 0; j < result2.size(); ++j) {
            remoteView.showMat("Plain best " + std::to_string(j), result2[j]);
        }

#endif

        auto end = std::chrono::high_resolution_clock::now();

        double time = ((double) (end - start).count()) / 1e6;
        double time2 = ((double) (endDisp - startDisp).count()) / 1e6;
        double avgA = 2. / ((i < 50 ? 5 : 50) + 1);
        avgTime = avgTime == 0. ? time : avgA * time + (1 - avgA) * avgTime;
        avgTime2 = avgTime2 == 0. ? time2 : avgA * time2 + (1 - avgA) * avgTime2;
        avgFps = avgFps == 0. ? fps : avgA * fps + (1 - avgA) * avgFps;

        std::cerr << "fps " << avgFps << " time " << time << " time2 " << time2 << " load " << avgTime * 100 / ((double)us / 1e6) << " %, avg " << avgTime << " %, avg2 " << avgTime2 << std::endl;

        sendingLock.lock();
        tm << "PERF " << fps << " " << time << std::endl;
        sendingLock.unlock();

#ifdef HAVE_OPENCV_HIGHGUI

        if (cv::waitKey(1) >= 0) {
            break;
        }
#endif
    }

    std::cerr << "exiting..." << std::endl;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::destroyAllWindows();
#endif

    captureServer.stop();
    streamingServer.stop();
    tmServer.stop();
    commandServer.stop();

    captureThread.join();
    streamingServerThread.join();
    tmServerThread.join();
    commandServerThread.join();

    return 0;
}
