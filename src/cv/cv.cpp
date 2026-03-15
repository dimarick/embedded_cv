#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif
#include <iostream>
#include <chrono>
#include <cpptrace/cpptrace.hpp>
#include <IpcServer.h>
#include <thread>
#include <atomic>
#include <SocketFactory.h>
#include <csignal>
#include <core/opencl/ocl_defs.hpp>
#include <core/ocl.hpp>
#include <common/RemoteView.h>

#include "DisparityEvaluator.h"
#include "capture/SocketCapture.h"
#include "common/MatStorage.h"
#include "common/Telemetry.h"

using namespace mini_server;

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

static std::atomic running = true;

int main(int argc, const char **argv) {
    cpptrace::register_terminate_handler();
    std::cerr << "Built with OpenCV " << CV_VERSION << std::endl;
    auto remoteView = ecv::RemoteView("/tmp/cv_stream");

    if (argc < 2) {
        std::cout << "too less command arguments";

        return -1;
    }

    cv::ocl::setUseOpenCL(true);

    std::shared_ptr<IpcServer> tmServer;
    IpcServer commandServer;

    commandServer.setSocket(SocketFactory::createServerSocket("/tmp/cv_ctl", 1));
    commandServer.setOnMessage([&remoteView, &commandServer] (int socket, const std::string &message) {
        std::istringstream is(message);
        std::ostringstream os;
        char c;
        std::string command;
        is >> c;
        assert(c == '[');
        is >> std::quoted(command);
        if (command == "GET_VIEWS") {
            os << std::quoted("VIEWS") << "{";
            auto views = remoteView.getViews();
            for (const auto &[viewName, view] : views) {
                if (viewName != views.begin()->first) os << ",";
                os << std::quoted(viewName) << ":" << "{";
                    os << "type" << ":" << view.type << ",",
                    os << "codec" << ":" << view.codec << ",",
                    os << "channels" << ":" << view.channels << ",",
                    os << "w" << ":" << view.w << ",",
                    os << "h" << ":" << view.h,
                os << "}";
            }
            os << "}";
            commandServer.send(socket, os.str(), IpcServer::MessageTypeEnum::TYPE_CONTROL);
            ecv::Telemetry::debug(std::format("command {} processed with reply {}", message, os.str()));
        } else if (command == "MOVE") {
            std::string direction;
            is >> std::quoted(direction);
            os << "[" << std::quoted("MOVING") << "," << std::quoted(direction) << "]";

            commandServer.send(socket, os.str(), IpcServer::MessageTypeEnum::TYPE_CONTROL);
            ecv::Telemetry::debug(std::format("command {} processed with reply {}", message, os.str()));
        } else if (command == "MOVE_TO") {
            int x, y;
            is >> x;
            is >> y;
            os << "[" << std::quoted("MOVING_TO") << "," << x <<"," << y << "]";

            commandServer.send(socket, os.str(), IpcServer::MessageTypeEnum::TYPE_CONTROL);
            ecv::Telemetry::debug(std::format("command {} processed with reply {}", message, os.str()));
        } else {
            ecv::Telemetry::error(std::format("command {} is not supported", command));
        }
    });

    ecv::Telemetry::setLogLevel(ecv::Telemetry::DEBUG);
    tmServer = std::make_shared<IpcServer>();
    tmServer->setSocket(SocketFactory::createServerSocket("/tmp/cv_tm", 10));
    ecv::Telemetry::setServer(tmServer);

    ecv::SocketCapture socketCapture(argv[1]);
    while (!socketCapture.run()) {
        ecv::Telemetry::error(std::format("Failed to connect to {}. Error is {}", argv[1], strerror(errno)));
        sleep(5);
    }

    signal(SIGINT, [](int signal) {
        running = false;
    });

    std::thread commandServerThread = std::thread([&commandServer]() {
        commandServer.serve();
    });

    std::thread tmServerThread = std::thread([&tmServer]() {
        tmServer->serve();
    });

    double fps = 0.;
    double avgFps = 0.;
    double avgTime = 0.;
    double avgTime2 = 0.;

#ifndef HAVE_OPENCL
    throw std::runtime_error("OpenCV built without opencl");
#endif

    if (!cv::ocl::haveOpenCL()) {
        throw std::runtime_error("OpenCL is not available");
    }

    // std::vector<cv::Mat> invMaps;
    std::vector<cv::UMat> maps;

    for (int i = 0; i < 10; i++) {
        cv::UMat map;
        if (!ecv::MatStorage::matRead(std::format("map{}.bin", i), map)) {
            break;
        }
        maps.emplace_back(map);
    }

    // for (int i = 0; i < maps.size(); ++i) {
    //     invMap(maps[i].getMat(cv::ACCESS_READ), invMaps[i]);
    // }

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
    cv::FileStorage fs;
    cv::FileStorage fs_write;
    auto clahe = cv::createCLAHE(1, cv::Size(3,3));

    auto prev = std::chrono::high_resolution_clock::now();

    for (int i = 0; running; i++) {
        std::vector<ecv::CaptureInfo> captureInfo;
        auto frames = socketCapture.getNewFrames(&captureInfo);

        if (frames.empty()) {
            continue;
        }

        std::vector<cv::UMat> result(frames.size());

        auto start = std::chrono::high_resolution_clock::now();
        std::chrono::system_clock::time_point startDisp;
        std::chrono::system_clock::time_point endDisp;

        cv::rotate(frames[1], frames[1], cv::ROTATE_180);

        auto now = std::chrono::high_resolution_clock::now();
        auto us = (double) (now - prev).count();
        prev = now;

        fps = 1e9 / us;

        if (frames[0].empty()) {
            break;
        }

        for (int j = 0; j < frames.size(); ++j) {
            remoteView.showMat(std::format("Source {}", j), frames[j], captureInfo[j].created_at);
        }


// #pragma omp parallel default(none) shared(frames, remoteView, result, maps)
        for (int j = 0; j < frames.size(); ++j) {
            cv::remap(frames[j], result[j], maps[j], cv::noArray(), cv::INTER_NEAREST);
        }

        for (int j = 0; j < frames.size(); ++j) {
            remoteView.showMat(std::format("Plain {}", j), frames[j]);
        }

#ifdef HAVE_OPENCV_HIGHGUI
        std::vector<cv::Mat> result2(result.size());
        for (auto j = 0; j < result2.size(); j++) {
            result[j].getMat(cv::ACCESS_RW).copyTo(result2[j]);
        }
        cv::drawMarker(frames[0], mouseSrc, cv::Scalar(255, 0, 255), cv::MarkerTypes::MARKER_CROSS, 30, 2);
        // auto plainMap1 = invMaps[0].ptr<cv::Point2f>(mouseSrc.y, mouseSrc.x);
        // auto plainMap2 = maps[0].getMat(cv::ACCESS_READ).ptr<cv::Point2f>((int)plainMap1->y, (int)plainMap1->x);
        // cv::drawMarker(result2[0], cv::Point2i((int) plainMap1->x, (int) plainMap1->y), cv::Scalar(0, 255, 0), cv::MarkerTypes::MARKER_TILTED_CROSS, 30, 2);
        // cv::drawMarker(frames[1], cv::Point2i((int) plainMap2->x, (int) plainMap2->y), cv::Scalar(0, 255, 0), cv::MarkerTypes::MARKER_CROSS, 30, 2);
        // cv::drawMarker(result2[1], cv::Point2i((int) plainMap1->x, (int) plainMap1->y), cv::Scalar(0, 255, 0), cv::MarkerTypes::MARKER_TILTED_CROSS, 30, 2);

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

    tmServer->stop();
    commandServer.stop();

    tmServerThread.join();
    commandServerThread.join();
    socketCapture.stop();

    return 0;
}
