#ifndef EMBEDDED_CV_MULTICAMERACALIBRATION_H
#define EMBEDDED_CV_MULTICAMERACALIBRATION_H

#include <queue>
#include <thread>
#include <condition_variable>
#include <utility>
#include <shared_mutex>
#include "opencv2/core.hpp"
#include "CalibrateFrameCollector.h"
#include "Calibrator.h"
#include "GridPreferredSizeProvider.h"
#include "CalibrateMapper.h"

namespace ecv {

    class MultiCameraCalibration {
    private:
        static const int TRAIN_SAMPLE_SIZE = 50;
        static const int VALIDATE_SAMPLE_SIZE = 100;

        typedef std::vector<CalibrateFrameCollector::FrameRef> FrameRefList;

        const int numCameras;
        const cv::Size frameSize;

        GridPreferredSizeProvider &gridPreferredSizeProvider;
        CalibrateFrameCollector &frameCollector;

        double temperature = 100.;
        double annealFreq = 0.5;
        double multicamCosts = 1. / 0.;
        std::vector<double> gridDistanceCosts;
        std::vector<double> viewMulticamCosts;
        mutable std::shared_mutex dataMutex;
        std::vector<CalibrationData> data;
        std::vector<CalibrationData> rectificationData;
        std::vector<cv::Mat> rectifiedMap;
        std::function<void(int cameraId, const MultiCameraCalibration &that)> onUpdateCallback;

        [[nodiscard]] std::vector<cv::Mat> getObjectPointsFromFrameSets(const std::vector<FrameRefList> &frameSets) const;

        void printMulticamCalibrationStats(const std::vector<cv::Mat> &camMatrices, const std::vector<cv::Mat> &Rs,
                                           const std::vector<cv::Mat> &Ts, const std::string &unit);
    public:
        explicit MultiCameraCalibration(
                const int numCameras,
                const cv::Size frameSize,
                std::function<void(int cameraId, const MultiCameraCalibration &that)> onUpdateCallback,
                GridPreferredSizeProvider &gridPreferredSizeProvider,
                CalibrateFrameCollector &frameCollector
        ) :
                numCameras(numCameras),
                frameSize(frameSize),
                onUpdateCallback(std::move(onUpdateCallback)),
                gridPreferredSizeProvider(gridPreferredSizeProvider),
                frameCollector(frameCollector) {

            gridDistanceCosts.reserve(numCameras);
            viewMulticamCosts.reserve(numCameras);
            data.reserve(numCameras);
            rectificationData.reserve(numCameras);
            rectifiedMap.reserve(numCameras);
            for (int i = 0; i < numCameras; ++i) {
                gridDistanceCosts.emplace_back(1. / 0.);
                viewMulticamCosts.emplace_back();
                data.emplace_back(frameSize);
                rectificationData.emplace_back(frameSize);
                rectifiedMap.emplace_back();
            }
        }

        void multicamThreadCallback(const std::vector<FrameRefList> &pendingFramesPerCam);
        bool isValid(const FrameRefList &frameSet) const;
        std::vector<std::vector<cv::Mat>> getImagePointsFromFrameSets(const std::vector<FrameRefList> &frameSets) const;

        [[nodiscard]] CalibrationData getCalibrationData(int cameraId) const {
            CalibrationData result;
            {
                std::lock_guard lock(dataMutex);
                result = data[cameraId];
            }

            return result;
        }

        void setCalibrationData(int cameraId, const CalibrationData &updatedData) {
            std::unique_lock lock(dataMutex);
            data[cameraId] = updatedData;
        }

        [[nodiscard]] CalibrationData getRectificationData(int cameraId) const {
            CalibrationData result;
            {
                std::lock_guard lock(dataMutex);
                result = rectificationData[cameraId];
            }

            return result;
        }

        void setRectificationData(int cameraId, const CalibrationData &updatedData) {
            std::unique_lock lock(dataMutex);
            rectificationData.at(cameraId) = updatedData;
        }

        [[nodiscard]] cv::Mat getRectifiedMap(int cameraId) const {
            return rectifiedMap[cameraId];
        }

        [[nodiscard]] double getViewMulticamCosts(int cameraId) const {
            return viewMulticamCosts[cameraId];
        }

        [[nodiscard]] double getAlignedBias(int cameraId) const {
            return gridDistanceCosts[cameraId];
        }

        double verifyParamsUsingGridMatch(const std::vector<cv::Mat> &imagePoints,
                                          const CalibrationData &calibrationData) const;

        static void rectifyImagePoints(const std::vector<ecv::CalibrateMapper::Point3> &imagePoints,
                                std::vector<ecv::CalibrateMapper::Point3> &plainPoints,
                                const CalibrationData &calibrationData);

        static void rectifyImagePoints(const std::vector<cv::Mat> &imagePoints, std::vector<cv::Mat> &plainPoints,
                                const CalibrationData &calibrationData);
    };

} // ecv

#endif //EMBEDDED_CV_MULTICAMERACALIBRATION_H
