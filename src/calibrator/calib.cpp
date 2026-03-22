#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <chrono>
#include <csignal>
#include <cpptrace/cpptrace.hpp>
#include <IpcServer.h>
#include <thread>
#include <SocketFactory.h>
#include <core/ocl.hpp>
#include "Calibrator.h"
#include "CalibrateMapper.h"
#include "CalibrateFrameCollector.h"
#include "CalibrationStrategy.h"
#include <common/RemoteView.h>
#include <common/Telemetry.h>

#include "capture/SocketCapture.h"
#include "common/MatStorage.h"

static std::atomic running = true;

int main(int argc, const char **argv) {
    cpptrace::register_terminate_handler();

    std::cerr << (cv::ocl::haveOpenCL() ? "with OpenCl" : "cpu only") << std::endl;

    std::cerr << "Built with OpenCV " << CV_VERSION << ", cv::ocl::haveSVM(): " << cv::ocl::haveSVM() << std::endl;

    if (argc < 2) {
        std::cout << "too less command arguments";

        return -1;
    }

    cv::ocl::setUseOpenCL(true);

    if (!cv::ocl::useOpenCL()) {
        throw std::runtime_error("OCL not available");
    }

    signal(SIGINT, [](int signal) {
        running = false;
    });

    auto remoteView = ecv::RemoteView("/tmp/cv_calib_stream");

    auto tm = std::make_shared<mini_server::IpcServer>();
    tm->setSocket(mini_server::SocketFactory::createServerSocket("/tmp/cv_calib_tm", 10));
    std::thread tmThread([&tm] () {
        tm->serve();
    });

    ecv::Telemetry::setServer(tm);
    ecv::Telemetry::setLogLevel(ecv::Telemetry::LogLevel::DEBUG);

    ecv::SocketCapture socketCapture(argv[1]);
    socketCapture.run();

    std::vector<cv::Mat> frames;
    std::vector<cv::UMat> uFrames;
    do {
        frames = socketCapture.getNewFrames();
    } while (frames.empty());

    socketCapture.stop();

    const cv::Size &size = frames[0].size();
    std::vector<ecv::CalibrateMapper> calibrateMapper(frames.size()), testCalibrateMapper(frames.size());
    std::vector<ecv::Calibrator> calibrator(frames.size());
    std::vector calibrationData(frames.size(), ecv::CalibrationData(size));
    std::vector rectificationData(frames.size(), ecv::CalibrationData(size));

    std::vector bestQ(frames.size(), 1. / 0.);
    std::vector bestPairQ(frames.size(), 1. / 0.);
    std::vector bestRoiQ(frames.size(), 0.);
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
        cv::FileStorage storage;

        storage.open(std::format("config_{}.yaml", cameraId), cv::FileStorage::WRITE);

        if (storage.isOpened()) {
            rectificationData[cameraId].store(storage);
            storage.release();
        }

        ecv::MatStorage::matWrite(std::format("map_{}.bin", cameraId), rectifiedMaps[cameraId]);

        std::cout << "Calibration updated " << cameraId << ",\t" << that.getProgress(cameraId) << "%\tcost " << that.getViewCosts(cameraId) << std::endl;
    });
    mini_server::IpcServer commandServer;
    mini_server::IpcServer streamingServer;

    std::atomic calibrating = false;

    commandServer.setSocket(mini_server::SocketFactory::createServerSocket("/tmp/cv_calib_ctl", 1));
    commandServer.setOnMessage([&commandServer, &calibrating, &calibrationStrategy, &socketCapture] (int socket, const std::string &message) {
        std::istringstream is(message);
        std::ostringstream os;
        char c;
        std::string command;
        is >> c;
        assert(c == '[');
        is >> std::quoted(command);
        if (command == "RESET_DATASET") {
            commandServer.send(socket, "[\"RESET_DATASET_OK\"]", 0, mini_server::IpcServer::MessageTypeEnum::TYPE_CONTROL);
            ecv::Telemetry::debug(std::format("command {} processed with reply {}", message, os.str()));
        } else if (command == "RESET_PARAMS") {
            commandServer.send(socket, "[\"RESET_PARAMS_OK\"]", 0, mini_server::IpcServer::MessageTypeEnum::TYPE_CONTROL);
            ecv::Telemetry::debug(std::format("command {} processed with reply {}", message, os.str()));
        } else if (command == "SAVE_PARAMS") {
            commandServer.send(socket, "[\"SAVED_PARAMS_OK\"]", 0, mini_server::IpcServer::MessageTypeEnum::TYPE_CONTROL);
            ecv::Telemetry::debug(std::format("command {} processed with reply {}", message, os.str()));
        } else if (command == "START_CALIBRATION") {
            if (!calibrating) {
                calibrating = true;
                calibrationStrategy.runCalibration();
                socketCapture.run();
                commandServer.send(socket, "[\"STARTED_CALIBRATION\"]", 0, mini_server::IpcServer::MessageTypeEnum::TYPE_CONTROL);
                ecv::Telemetry::debug(std::format("command {} processed with reply {}", message, os.str()));
            }
        } else if (command == "STOP_CALIBRATION") {
            if (calibrating) {
                calibrating = false;
                socketCapture.stop();
                calibrationStrategy.stopCalibration();
                commandServer.send(socket, "[\"STOPPED_CALIBRATION\"]", 0, mini_server::IpcServer::MessageTypeEnum::TYPE_CONTROL);
                ecv::Telemetry::debug(std::format("command {} processed with reply {}", message, os.str()));
            }
        } else {
            ecv::Telemetry::error(std::format("command {} is not supported", command));
        }
    });

    std::thread commandServerThread = std::thread([&commandServer]() {
        commandServer.serve();
    });

    while (running) {
        if (!calibrating) {
            usleep(50000);
            continue;
        }

        std::vector<ecv::CaptureInfo> captureInfo;
        frames = socketCapture.getNewFrames(&captureInfo);

        if (frames.empty()) {
            continue;
        }

        uFrames.resize(frames.size());
#pragma omp parallel for default(none) shared(frames, uFrames)
        for (int i = 0; i < frames.size(); ++i) {
            frames[i].copyTo(uFrames[i]);
        }

        if (frames.empty()) {
            continue;
        }

        std::vector<cv::Mat> plainFrames(frames.size());
        std::vector<cv::Mat> rectifiedFrames(frames.size());
        std::vector<std::mutex> framesMutex(frames.size());
        std::vector<cv::Mat> readingFrames(frames.size());
        std::vector<cv::Mat> readFrames(frames.size());

        std::vector peaks(frames.size(), std::vector<ecv::CalibrateMapper::Point3>(500));
#pragma omp parallel for default(none) shared(peaks, uFrames, calibrateMapper)
        for (int i = 0; i < uFrames.size(); ++i) {
            size_t peaksSize;

            calibrateMapper[i].detectPeaks(uFrames[i], peaks[i], &peaksSize);
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

            auto frameQuality = calibrateMapper[i].detectFrameImagePointsGrid(frames[i].size(), peaks[i], imageGrid, &w, &h, debug);

            std::vector<ecv::CalibrateMapper::Point3> calibImageGrid(500), calibObjectGrid(500);

            if (i == 1) {
                ecv::Telemetry::status("calibration", "grid.error.raw", std::format("Grid error is {}", frameQuality));
            }

            if (frameQuality < 1 && w >= 6 && h >= 3) {
                calibrateMapper[i].generateFrameObjectPointsGrid(objectGrid, w, h);
                imageGrid.resize(w * h);
                objectGrid.resize(w * h);

                frameSet[i] = calibrationStrategy.createFrame(i, imageGrid, objectGrid, (int)w, (int)h, frameQuality, captureInfo[i].created_at);
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

            ecv::Telemetry::status("calib", std::format("cam.{}.pos.text", i), "cam.0.pos.text");
            ecv::Telemetry::status("calib", std::format("cam.{}.dataset.size", i), calibrationStrategy.getFrameCount(i));

            auto cam1 = calibrationDatum;

            cv::Vec3d t;
            if (cam1.Ts.rows == 3 && cam1.Ts.cols == 1) {
                t = cv::Vec3d(cam1.Ts.at<double>(0,0), cam1.Ts.at<double>(1,0), cam1.Ts.at<double>(2,0));
            } else if (cam1.Ts.rows == 1 && cam1.Ts.cols == 3) {
                t = cv::Vec3d(cam1.Ts.at<double>(0,0), cam1.Ts.at<double>(0,1), cam1.Ts.at<double>(0,2));
            } else {
                t = cv::Vec3d(1. / 0., 1. / 0., 1. / 0.);
            }
            auto baseline = cv::norm(t);

            cv::Vec3d rvec = !calibrationDatum.R.empty() ? calibrationDatum.R : cv::Vec3d(1. / 0., 1. / 0., 1. / 0.);
            double angle = cv::norm(rvec);
            double angle_deg = angle * 180.0 / CV_PI;
            cv::Vec3d axis = rvec / (angle + 1e-12);

            ecv::Telemetry::status("calib", {
                std::format("cam.{}.multicam_repr_error", i),
                std::format("cam.{}.repr_error", i),
                std::format("cam.{}.dataset.size", i),
                std::format("cam.{}.dataset.percent", i),
                std::format("cam.{}.aligned_bias", i),
                std::format("cam.{}.T.baseline", i),
                std::format("cam.{}.T.x", i),
                std::format("cam.{}.T.y", i),
                std::format("cam.{}.T.z", i),
                std::format("cam.{}.R.angle", i),
                std::format("cam.{}.R.axis.x", i),
                std::format("cam.{}.R.axis.y", i),
                std::format("cam.{}.R.axis.z", i),
                std::format("cam.{}.R.x", i),
                std::format("cam.{}.R.y", i),
                std::format("cam.{}.R.z", i),
                std::format("cam.{}.Fx", i),
                std::format("cam.{}.Сx", i),
                std::format("cam.{}.Cy", i)
            }, std::vector<double>{
                calibrationStrategy.getViewMulticamCosts(0),
                calibrationStrategy.getViewCosts(i),
                (double)calibrationStrategy.getFrameCount(i),
                calibrationStrategy.getProgress(i),
                calibrationStrategy.getAlignedBias(i),
                baseline,
                t[0],
                t[1],
                t[2],
                angle_deg,
                axis[0],
                axis[1],
                axis[2],
                rvec[0] * 180.0 / CV_PI,
                rvec[1] * 180.0 / CV_PI,
                rvec[2] * 180.0 / CV_PI,
                calibrationStrategy.getF(i).x,
                calibrationStrategy.getC(i).x,
                calibrationStrategy.getC(i).y
            });

#ifdef HAVE_OPENCV_HIGHGUI
            if (!plainFrames[i].empty()) {
                int height = plainFrames[i].rows;
                int step = height / (10 + 1);
                for (int j = 1; j <= 10; ++j) {
                    int y = j * step;
                    cv::line(rectifiedFrames[i], cv::Point(0, y), cv::Point(rectifiedFrames[i].cols - 1, y), cv::Scalar(0, 255, 0), 1);
                }

                remoteView.showMat(std::format("Flat {}", i), plainFrames[i]);
            }
            if (!rectifiedFrames[i].empty()) {
                remoteView.showMat(std::format("Aligned {}", i), rectifiedFrames[i]);
            }

            if (!debug.empty()) {
                remoteView.showMat(std::format("Source {}", i), debug);
            }
#endif
        }

        calibrationStrategy.addFrameSet(frameSet);

        ecv::Telemetry::status("calib", {
            "multicam.count",
            "multicam.dataset.size",
        }, {
            std::to_string(frames.size()),
            std::to_string(calibrationStrategy.getFrameSetCount(0)),
        });

#ifdef HAVE_OPENCV_HIGHGUI
        if (remoteView.waitKey() != -1) {
            break;
        }
#endif
    }

    calibrationStrategy.stopCalibration();
    tm->stop();
    commandServer.stop();
    streamingServer.stop();
    socketCapture.stop();

    tmThread.join();
    commandServerThread.join();

    return 0;
}
