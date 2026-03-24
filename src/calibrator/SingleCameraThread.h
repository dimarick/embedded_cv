#ifndef EMBEDDED_CV_SINGLECAMERATHREAD_H
#define EMBEDDED_CV_SINGLECAMERATHREAD_H

#include <atomic>
#include <shared_mutex>
#include <utility>
#include "opencv2/core.hpp"
#include "CalibrateFrameCollector.h"
#include "Calibrator.h"
#include "GridPreferredSizeProvider.h"
#include "CalibrateMapper.h"

namespace ecv {
    class SingleCameraThread {
    public:
        static constexpr int TRAIN_SAMPLE_SIZE = 50;
        static constexpr int VALIDATE_SAMPLE_SIZE = 100;
        typedef std::vector<CalibrateFrameCollector::FrameRef> FrameRefList;

    private:
        static GridPreferredSizeProvider nullGridPreferredSizeProvider;
        static CalibrateFrameCollector nullFrameCollector;
        static Calibrator nullCalibrator;

        const int cameraId;
        const cv::Size frameSize;
        std::function<void(const SingleCameraThread &that)> onUpdateCallback;

        GridPreferredSizeProvider &gridPreferredSizeProvider;
        CalibrateFrameCollector &frameCollector;
        const Calibrator &calibrator;

        CalibrationData data;
        CalibrationData tmpData;
        cv::Mat map;

        double reprCost = 1. / 0.;
        double gridMaxCost = 1. / 0.;
        double viewCost = 1. / 0.;
        double failCount = 0;
        double temperature = 10.;

        std::atomic<bool> running = false;
        mutable std::shared_mutex dataMutex;
    public:
        explicit SingleCameraThread() :
            cameraId(-1),
            frameSize(cv::Size(0, 0)),
            onUpdateCallback(nullptr),
            gridPreferredSizeProvider(nullGridPreferredSizeProvider),
            frameCollector(nullFrameCollector),
            calibrator(nullCalibrator)
        {
        }

        explicit SingleCameraThread(
            int cameraId,
            cv::Size frameSize,
            std::function<void(const SingleCameraThread &that)> onUpdateCallback,
            GridPreferredSizeProvider &gridPreferredSizeProvider,
            CalibrateFrameCollector &frameCollector,
            const Calibrator &calibrator
        ) :
            cameraId(cameraId),
            frameSize(frameSize),
            onUpdateCallback(std::move(onUpdateCallback)),
            gridPreferredSizeProvider(gridPreferredSizeProvider),
            frameCollector(frameCollector),
            calibrator(calibrator) {}

        void camThreadCallback(const FrameRefList &frames);

        void reset() {
            std::unique_lock lock(dataMutex);
            data = CalibrationData();
            tmpData = CalibrationData();

            reprCost = 1. / 0.;
            gridMaxCost = 1. / 0.;
            viewCost = 1. / 0.;
        }

        static void undistortImagePoints(const std::vector<ecv::CalibrateMapper::Point3> &imagePoints, std::vector<ecv::CalibrateMapper::Point3> &plainPoints, const CalibrationData &calibrationData);
        static void undistortImagePoints(const std::vector<cv::Mat> &imagePoints, std::vector<cv::Mat> &plainPoints, const CalibrationData &calibrationData);

        [[nodiscard]] CalibrationData getCalibrationData() const {
            CalibrationData result;
            {
                std::lock_guard lock(dataMutex);
                result = data;
            }

            return result;
        }

        [[nodiscard]] cv::Point2d getF() {
            std::lock_guard lock(dataMutex);
            return {
                    data.cameraMatrix.at<double>(0, 0),
                    data.cameraMatrix.at<double>(1, 1),
            };
        }

        [[nodiscard]] cv::Point2d getC() {
            std::lock_guard lock(dataMutex);
            return {
                    data.cameraMatrix.at<double>(0, 2),
                    data.cameraMatrix.at<double>(1, 2),
            };
        }

        void setCalibrationData(const CalibrationData &updatedData) {
            std::unique_lock lock(dataMutex);
            data = updatedData;
        }

        [[nodiscard]] cv::Mat getMap() const {
            std::lock_guard lock(dataMutex);
            return map;
        }

        void setMap(const cv::Mat &m) {
            std::lock_guard lock(dataMutex);
            map = m.clone();
        }

        [[nodiscard]] double getProgress() const {
            return frameCollector.getProgress();
        }

        [[nodiscard]] size_t getFrameCount() const {
            return frameCollector.getFrameCount();
        }

        [[nodiscard]] size_t getFrameSetCount() const {
            return frameCollector.getFrameSetCount();
        }

        [[nodiscard]] double getViewCosts() const {
            return viewCost;
        }
    };
};

#endif //EMBEDDED_CV_SINGLECAMERATHREAD_H
