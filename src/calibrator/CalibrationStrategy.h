#ifndef EMBEDDED_CV_CALIBRATIONSTRATEGY_H
#define EMBEDDED_CV_CALIBRATIONSTRATEGY_H

#include <queue>
#include <thread>
#include <condition_variable>
#include <utility>
#include <shared_mutex>
#include "opencv2/core.hpp"
#include "CalibrateFrameCollector.h"
#include "Calibrator.h"
#include "GridPreferredSizeProvider.h"

static const int MAX_FRAMES_QUEUE = 100;
static const int SAMPLE_SIZE = 20;
static const int VALIDATE_SAMPLE_SIZE = 200;
namespace ecv {

    class CalibrationStrategy {
    public:
        typedef std::vector<CalibrateFrameCollector::FrameRef> FrameRefList;
    private:
        struct FrameSetCompare {
            bool operator()(const FrameRefList &a, const FrameRefList &b) const {
                double costA = 0, costB = 0;

                for (int i = 0; i < a.size(); ++i) {
                    if (a[i] != nullptr) {
                        costA += a[i]->cost;
                    }
                    if (b[i] != nullptr) {
                        costB += b[i]->cost;
                    }
                }

                if (costA == costB) return true;
                return costA > costB;
            }
        };
        struct FrameCompare {
            bool operator()(const CalibrateFrameCollector::FrameRef &a, const CalibrateFrameCollector::FrameRef &b) const {
                if (a == nullptr && b == nullptr) return true;
                if (a == nullptr) return false;
                if (b == nullptr) return true;
                if (a->cost == b->cost) return true;
                return a->cost > b->cost;
            }
        };

        int numCameras = 0;
        cv::Size frameSize;
        volatile bool running = false;
        GridPreferredSizeProvider gridPreferredSizeProvider;
        std::vector<CalibrateFrameCollector> frameCollectors;
        std::vector<ecv::Calibrator> calibrators;
        mutable std::shared_mutex dataMutex;
        std::vector<Calibrator::CalibrationData> data;
        std::vector<cv::Mat> map;
        std::vector<double> costs;
        double multicamCosts = 1. / 0.;
        double multicamRoiSize = 0.;
        std::mutex pendingFramesMutex;
        std::vector<std::set<CalibrateFrameCollector::FrameRef, FrameCompare>> pendingFrames;
        std::set<FrameRefList, FrameSetCompare> pendingFrameSets;
        std::vector<cv::FileStorage> frameDataStorage;
        std::condition_variable camThreadsWait;
        std::vector<std::thread> camThreads;
        std::condition_variable multicamThreadWait;
        std::thread multicamThread;
        std::function<void(int cameraId, const CalibrationStrategy &that)> onUpdateCallback;

        void camThreadCallback(const FrameRefList &pendingFrames, int cameraId);
        void multicamThreadCallback(const std::vector<FrameRefList> &pendingFramesPerCam);
        [[nodiscard]] std::vector<cv::Mat> getObjectPointsFromFrameSets(const std::vector<FrameRefList> &frameSets) const;

        void printMulticamCalibrationStats(const std::vector<cv::Mat> &camMatrices, const std::vector<cv::Mat> &Rs,
                                           const std::vector<cv::Mat> &Ts, const std::string &unit);
        std::vector<bool> findOutliersPerFrameError(const cv::Mat &errors, double k = 2.0);
        int filterOutliers(std::vector<FrameRefList> &frameSets, const std::vector<bool> &outliers) const;
    public:
        explicit CalibrationStrategy(
                cv::Size frameSize, int numCameras,
                std::function<void(int cameraId, const CalibrationStrategy &that)> onUpdateCallback
        ) : frameSize(frameSize), numCameras(numCameras), onUpdateCallback(std::move(onUpdateCallback)) {
            for (int i = 0; i < numCameras; ++i) {
                frameCollectors.emplace_back(frameSize);
                calibrators.emplace_back();
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

                frameCollectors[i].load(gridPreferredSizeProvider, frameDataStorage[i]);

                frameDataStorage[i].release();
            }
        }

        void loadConfig();

        [[nodiscard]] Calibrator::CalibrationData getCalibrationData(int cameraId) const {
            Calibrator::CalibrationData result;
            {
                std::shared_lock lock(dataMutex);
                result = data[cameraId];
            }

            return result;
        }

        void setCalibrationData(int cameraId, const Calibrator::CalibrationData &updatedData) {
            std::unique_lock lock(dataMutex);
            data[cameraId] = updatedData;
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

        CalibrateFrameCollector::FrameRef
        createFrame(int cameraId, const std::vector<cv::Point3d> &imagePoints,
                    const std::vector<cv::Point3d> &objectPoints,
                    int w, int h, double cost, double ts);

        void addFrameSet(const FrameRefList &frameSet);

        void runCalibration();

        void stopCalibration() noexcept;

        ~CalibrationStrategy() {
            stopCalibration();
        }

        std::vector<std::vector<cv::Mat>>
        getImagePointsFromFrameSets(const std::vector<FrameRefList> &frameSets) const;

        bool isValid(const FrameRefList &frameSet) const;
    };
}
#endif //EMBEDDED_CV_CALIBRATIONSTRATEGY_H
