#ifndef EMBEDDED_CV_SINGLECAMERATHREAD_H
#define EMBEDDED_CV_SINGLECAMERATHREAD_H

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
    class SingleCameraThread {
    public:
        static const int TRAIN_SAMPLE_SIZE = 50;
        static const int VALIDATE_SAMPLE_SIZE = 100;
        typedef std::vector<CalibrateFrameCollector::FrameRef> FrameRefList;
    private:
        static GridPreferredSizeProvider nullGridPreferredSizeProvider;
        static CalibrateFrameCollector nullFrameCollector;
        static Calibrator nullCalibrator;

        const int cameraId;
        const cv::Size frameSize;
        const std::function<void(const SingleCameraThread &that)> onUpdateCallback;

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
        void undistortImagePoints(const std::vector<cv::Mat> &imagePoints, std::vector<cv::Mat> &plainPoints, const CalibrationData &calibrationData) const;
        void undistortImagePoints(const std::vector<ecv::CalibrateMapper::Point3> &imagePoints, std::vector<ecv::CalibrateMapper::Point3> &plainPoints, const CalibrationData &calibrationData) const;
        void converPoints(const cv::Mat &pp, std::vector<ecv::CalibrateMapper::Point3> &points) const;
        void converPoints(const std::vector<ecv::CalibrateMapper::Point3> &points, cv::Mat &pp) const;
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
            const std::function<void(const SingleCameraThread &that)> &onUpdateCallback,
            GridPreferredSizeProvider &gridPreferredSizeProvider,
            CalibrateFrameCollector &frameCollector,
            const Calibrator &calibrator
        ) :
            cameraId(cameraId),
            frameSize(frameSize),
            onUpdateCallback(onUpdateCallback),
            gridPreferredSizeProvider(gridPreferredSizeProvider),
            frameCollector(frameCollector),
            calibrator(calibrator)
        {

        }

        void camThreadCallback(const FrameRefList &frames);

        [[nodiscard]] CalibrationData getCalibrationData() const {
            CalibrationData result;
            {
                std::shared_lock lock(dataMutex);
                result = data;
            }

            return result;
        }

        [[nodiscard]] cv::Point2d getF() {
            std::shared_lock lock(dataMutex);
            return {
                    data.cameraMatrix.at<double>(0, 0),
                    data.cameraMatrix.at<double>(1, 1),
            };
        }

        [[nodiscard]] cv::Point2d getC() {
            std::shared_lock lock(dataMutex);
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
            std::shared_lock lock(dataMutex);
            return map;
        }

        void setMap(const cv::Mat &m) {
            std::shared_lock lock(dataMutex);
            map = m.clone();
        }

        [[nodiscard]] double getProgress() const {
            return frameCollector.getProgress();
        }

        [[nodiscard]] auto getGridSize() const {
            return gridPreferredSizeProvider.getGridPreferredSize();
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
