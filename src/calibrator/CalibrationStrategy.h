#ifndef EMBEDDED_CV_CALIBRATIONSTRATEGY_H
#define EMBEDDED_CV_CALIBRATIONSTRATEGY_H

#include <queue>
#include <thread>
#include <condition_variable>
#include "opencv2/core.hpp"
#include "CalibrateFrameCollector.h"
#include "Calibrator.h"

static const int MAX_FRAMES_QUEUE = 10;
static const int SAMPLE_SIZE = 60;
namespace ecv {

    class CalibrationStrategy {
    private:
        struct FrameCompare {
            bool operator()(const CalibrateFrameCollector::FrameRef &a, const CalibrateFrameCollector::FrameRef &b) const {
                if (a->cost == b->cost) return true;
                return a->cost > b->cost;
            }
        };

        struct TsCompare {
            using is_transparent = void;
            bool operator()(const CalibrateFrameCollector::FrameRef &a, const CalibrateFrameCollector::FrameRef &b) const { return a->ts < b->ts; }
            bool operator()(const CalibrateFrameCollector::FrameRef& a, double v) const { return a->ts < v; }
            bool operator()(double v, const CalibrateFrameCollector::FrameRef& b) const { return v < b->ts; }
        };

        int numCameras = 0;
        cv::Size frameSize;
        volatile bool running = false;
        std::vector<CalibrateFrameCollector> frameCollectors;
        std::vector<ecv::Calibrator> calibrators;
        std::vector<Calibrator::CalibrationData> data;
        std::vector<cv::Mat> map;
        std::vector<double> costs;
        double multicamCosts = 1. / 0.;
        double multicamRoiSize = 0.;
        std::mutex pendingFramesMutex;
        std::vector<std::set<CalibrateFrameCollector::FrameRef>> pendingFrames;
        std::vector<std::set<CalibrateFrameCollector::FrameRef>> pendingFrames2;
        std::vector<cv::FileStorage> frameDataStorage;
        std::vector<std::condition_variable> camThreadsWait;
        std::vector<std::thread> camThreads;
        std::condition_variable multicamThreadWait;
        std::thread multicamThread;
        void (*onUpdateCallback)(int cameraId);

        void camThreadCallback(const std::vector<CalibrateFrameCollector::FrameRef> &pendingFrames, int cameraId);
        void multicamThreadCallback(const std::vector<std::set<CalibrateFrameCollector::FrameRef>> &pendingFramesPerCam);
        [[nodiscard]] double getFrameTimeInterval(const std::vector<std::set<CalibrateFrameCollector::FrameRef, TsCompare>> &framesPerCam) const;
        [[nodiscard]] CalibrateFrameCollector::FrameRef findClosestFrameByTs(const std::set<CalibrateFrameCollector::FrameRef, TsCompare> &set, double v) const;
    public:
        explicit CalibrationStrategy(cv::Size frameSize, int numCameras) : frameSize(frameSize), numCameras(numCameras) {
            for (int i = 0; i < numCameras; ++i) {
                frameCollectors.emplace_back(frameSize);
                data.emplace_back();
                map.emplace_back();
                costs.emplace_back(1. / 0.);
                frameDataStorage.emplace_back();
                camThreads.emplace_back();
            }
            pendingFrames.resize(numCameras);


            for (int i = 0; i < numCameras; ++i) {
                frameDataStorage[i].open(std::format("frameData{}.yaml", i), cv::FileStorage::READ);

                if (!frameDataStorage[i].isOpened()) {
                    continue;
                }

                frameCollectors[i].load(frameDataStorage[i]);

                frameDataStorage[i].release();
            }
        }

        [[nodiscard]] const Calibrator::CalibrationData &getCalibrationData(int cameraId) const {
            return data[cameraId];
        }

        [[nodiscard]] cv::Mat getMap(int cameraId) const {
            return map[cameraId];
        }

        [[nodiscard]] double getProgress(int cameraId) const {
            return frameCollectors[cameraId].getProgress();
        }

        [[nodiscard]] double getCosts(int cameraId) const {
            return costs[cameraId];
        }

        [[nodiscard]] double getMulticamCosts() const {
            return multicamCosts;
        }
        [[nodiscard]] double getMulticamRoiSize() const {
            return multicamRoiSize;
        }

        void addFrame(int cameraId,
                        const std::vector<cv::Point3d>& imagePoints,
                        const std::vector<cv::Point3d>& objectPoints,
                        int w, int h, double cost, double ts);

        void onCalibrationUpdated(void (*callback)(int cameraId)) {
            this->onUpdateCallback = callback;
        }

        void runCalibration();

        void stopCalibration();

        [[nodiscard]] bool isRunning() {
            return running;
        }

        std::vector<std::vector<CalibrateFrameCollector::FrameRef>>
        getFramePairs(const std::vector<std::set<CalibrateFrameCollector::FrameRef>> &framesPerCam);
    };
}
#endif //EMBEDDED_CV_CALIBRATIONSTRATEGY_H
