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
#include "CalibrateMapper.h"

static const int MAX_FRAMES_QUEUE = 100;
static const int TRAIN_SAMPLE_SIZE = 50;
static const int VALIDATE_SAMPLE_SIZE = 100;
namespace ecv {

    /**
     * Общая концепция: CalibrationStrategy принимает FrameSet - группа синхронизированных кадров с каждой камеры системы
     * Алгоритм работы:
     * Полученные кадры наполняют датасет, случайно распределяясь между набором тренировки и валидации
     * тренировочные кадры + некоторое количество из датасета участвуют в многократном инкрементальном обновлении параметров:
     * как внутренних (для каждой камеры в отдельности), так и внешних (взаимное расположение камер).
     * В основе этого обновления лежит алгоритм Левенберга-Марквардта, предоставленный библиотекой OpenCV.
     * Итеративно запускаем поступившие разделяем на валидацию и тренировку.
     * Тренировочные запускаем на калибровку. За исключением совсем плохих - добавляем в датасет.
     * Выполняем валидацию. Если ошибка уменьшилась, то принимаем результаты тренировки, иначе - отбрасываем.
     * Но в датасет они все равно попадают, если соответствуют критериям отбора.
     */
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
        std::atomic<bool> running = false;
        GridPreferredSizeProvider gridPreferredSizeProvider;
        std::vector<CalibrateFrameCollector> frameCollectors;
        std::vector<ecv::Calibrator> calibrators;
        mutable std::shared_mutex dataMutex;
        std::vector<Calibrator::CalibrationData> data;
        std::vector<Calibrator::CalibrationData> tmpData;
        std::vector<Calibrator::CalibrationData> rectificationData;
        std::vector<cv::Mat> map;
        std::vector<cv::Mat> rectifiedMap;
        std::vector<double> costs;
        std::vector<double> temperature;
        std::vector<double> annealFreq;
        double temperatureMc = 100.;
        double annealFreqMc = 0.5;
        double multicamCosts = 1. / 0.;
        std::vector<double> gridDistanceCosts;
        std::mutex pendingFramesMutex;
        std::vector<std::set<CalibrateFrameCollector::FrameRef, FrameCompare>> pendingFrames;
        std::set<FrameRefList, FrameSetCompare> pendingFrameSets;
        std::condition_variable camThreadsWait;
        std::vector<std::thread> camThreads;
        std::condition_variable multicamThreadWait;
        std::thread multicamThread;
        std::function<void(int cameraId, const CalibrationStrategy &that)> onUpdateCallback;
        std::string configPath;
        std::vector<double> viewCosts;
        std::vector<double> viewMulticamCosts;

        void camThreadCallback(const FrameRefList &pendingFrames, int cameraId);
        void multicamThreadCallback(const std::vector<FrameRefList> &pendingFramesPerCam);
        [[nodiscard]] std::vector<cv::Mat> getObjectPointsFromFrameSets(const std::vector<FrameRefList> &frameSets) const;

        void printMulticamCalibrationStats(const std::vector<cv::Mat> &camMatrices, const std::vector<cv::Mat> &Rs,
                                           const std::vector<cv::Mat> &Ts, const std::string &unit);
    public:
        explicit CalibrationStrategy(
                cv::Size frameSize, int numCameras,
                std::function<void(int cameraId, const CalibrationStrategy &that)> onUpdateCallback,
                std::string configPath = ""
        ) : frameSize(frameSize), numCameras(numCameras), configPath(std::move(configPath)), onUpdateCallback(std::move(onUpdateCallback)) {

            // Предварительно выделяем память, но не вызываем конструкторы объектов.
            // Это важно так как при добавлении второй камеры через emplace_back
            // происходит реаллокация и перемещение вектора.
            // Как правило, это происходит прозрачно и несет только накладные расходы на лишний вызов аллокатора,
            // но в частности FileStorage, (точнее порождаемые им FileNode)
            // содержат небезопасные ссылки (char * и т.п.), в результате FileStorage сваливается в SIGSEGV
            // на некоторых абсолютно корректных файлах
            frameCollectors.reserve(numCameras);
            calibrators.reserve(numCameras);
            data.reserve(numCameras);
            tmpData.reserve(numCameras);
            rectificationData.reserve(numCameras);
            map.reserve(numCameras);
            rectifiedMap.reserve(numCameras);
            costs.reserve(numCameras);
            camThreads.reserve(numCameras);
            viewCosts.reserve(numCameras);
            viewMulticamCosts.reserve(numCameras);
            temperature.reserve(numCameras);
            annealFreq.reserve(numCameras);

            for (int i = 0; i < numCameras; ++i) {
                frameCollectors.emplace_back(frameSize);
                calibrators.emplace_back();
                data.emplace_back(frameSize);
                tmpData.emplace_back(frameSize);
                rectificationData.emplace_back();
                map.emplace_back();
                rectifiedMap.emplace_back();
                costs.emplace_back(1. / 0.);
                gridDistanceCosts.emplace_back(1. / 0.);
                camThreads.emplace_back();
                viewCosts.emplace_back();
                viewMulticamCosts.emplace_back();
                temperature.emplace_back(100.);
                annealFreq.emplace_back(0.5);
            }
            pendingFrames.resize(numCameras);
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

        [[nodiscard]] Calibrator::CalibrationData getRectificationData(int cameraId) const {
            Calibrator::CalibrationData result;
            {
                std::shared_lock lock(dataMutex);
                result = rectificationData[cameraId];
            }

            return result;
        }

        [[nodiscard]] cv::Point2d getF(int cameraId) {
            return {
                data[cameraId].cameraMatrix.at<double>(0, 0),
                data[cameraId].cameraMatrix.at<double>(1, 1),
            };
        }

        [[nodiscard]] cv::Point2d getC(int cameraId) {
            return {
                data[cameraId].cameraMatrix.at<double>(0, 2),
                data[cameraId].cameraMatrix.at<double>(1, 2),
            };
        }

        void setCalibrationData(int cameraId, const Calibrator::CalibrationData &updatedData) {
            std::unique_lock lock(dataMutex);
            data.at(cameraId) = updatedData;
        }

        void setRectificationData(int cameraId, const Calibrator::CalibrationData &updatedData) {
            std::unique_lock lock(dataMutex);
            rectificationData.at(cameraId) = updatedData;
        }

        [[nodiscard]] cv::Mat getMap(int cameraId) const {
            return map[cameraId];
        }

        [[nodiscard]] cv::Mat getRectifiedMap(int cameraId) const {
            return rectifiedMap[cameraId];
        }

        [[nodiscard]] double getProgress(int cameraId) const {
            return frameCollectors[cameraId].getProgress();
        }

        [[nodiscard]] auto getGridSize() const {
            return gridPreferredSizeProvider.getGridPreferredSize();
        }

        [[nodiscard]] size_t getFrameCount(int cameraId) const {
            return frameCollectors[cameraId].getFrameCount();
        }

        [[nodiscard]] size_t getFrameSetCount(int cameraId) const {
            return frameCollectors[cameraId].getFrameSetCount();
        }

        [[nodiscard]] double getViewCosts(int cameraId) const {
            return viewCosts[cameraId];
        }

        [[nodiscard]] double getViewMulticamCosts(int cameraId) const {
            return viewMulticamCosts[cameraId];
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

        double
        verifyParamsUsingGridMatch(const std::vector<cv::Mat> &imagePoints,
                                   const Calibrator::CalibrationData &calibrationData) const;

        void undistortImagePoints(const std::vector<cv::Mat> &imagePoints, std::vector<cv::Mat> &plainPoints,
                                  const Calibrator::CalibrationData &calibrationData) const;

        void rectifyImagePoints(const std::vector<cv::Mat> &imagePoints, std::vector<cv::Mat> &plainPoints,
                                const Calibrator::CalibrationData &calibrationData) const;

        void rectifyImagePoints(const std::vector<ecv::CalibrateMapper::Point3> &imagePoints,
                                std::vector<ecv::CalibrateMapper::Point3> &plainPoints,
                                const Calibrator::CalibrationData &calibrationData) const;

        void undistortImagePoints(const std::vector<ecv::CalibrateMapper::Point3> &imagePoints,
                                  std::vector<ecv::CalibrateMapper::Point3> &plainPoints,
                                  const Calibrator::CalibrationData &calibrationData) const;

        void converPoints(const cv::Mat &pp, std::vector<ecv::CalibrateMapper::Point3> &points) const;

        void converPoints(const std::vector<ecv::CalibrateMapper::Point3> &points, cv::Mat &pp) const;
    };
}
#endif //EMBEDDED_CV_CALIBRATIONSTRATEGY_H
