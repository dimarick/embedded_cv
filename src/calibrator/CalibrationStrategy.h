#ifndef EMBEDDED_CV_CALIBRATIONSTRATEGY_H
#define EMBEDDED_CV_CALIBRATIONSTRATEGY_H

#include <thread>
#include <condition_variable>
#include <utility>
#include "opencv2/core.hpp"
#include "CalibrateFrameCollector.h"
#include "Calibrator.h"
#include "GridPreferredSizeProvider.h"
#include "CalibrateMapper.h"
#include "SingleCameraThread.h"
#include "MultiCameraCalibration.h"

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
        static const int MAX_FRAMES_QUEUE = 100;
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
        std::vector<std::shared_ptr<ecv::SingleCameraThread>> cameraThread;
        std::shared_ptr<MultiCameraCalibration> multicamThreadHandler;
        std::mutex pendingFramesMutex;
        std::vector<std::set<CalibrateFrameCollector::FrameRef, FrameCompare>> pendingFrames;
        std::set<FrameRefList, FrameSetCompare> pendingFrameSets;
        std::condition_variable camThreadsWait;
        std::vector<std::thread> camThreads;
        std::condition_variable multicamThreadWait;
        std::thread multicamThread;
        std::function<void(int cameraId, const CalibrationStrategy &that)> onUpdateCallback;
        std::string configPath;
    public:
        explicit CalibrationStrategy(
                cv::Size frameSize, int numCameras,
                std::function<void(int cameraId, const CalibrationStrategy &that)> onUpdateCallback,
                std::string configPath = ""
        ) :
        frameSize(frameSize),
        numCameras(numCameras),
        configPath(std::move(configPath)),
        onUpdateCallback(std::move(onUpdateCallback)) {

            // Предварительно выделяем память, но не вызываем конструкторы объектов.
            // Это важно так как при добавлении второй камеры через emplace_back
            // происходит реаллокация и перемещение вектора.
            // Как правило, это происходит прозрачно и несет только накладные расходы на лишний вызов аллокатора,
            // но в частности FileStorage, (точнее порождаемые им FileNode)
            // содержат небезопасные ссылки (char * и т.п.), в результате FileStorage сваливается в SIGSEGV
            // на некоторых абсолютно корректных файлах
            frameCollectors.reserve(numCameras);
            calibrators.reserve(numCameras);
            camThreads.reserve(numCameras);
            cameraThread.reserve(numCameras);
            for (int i = 0; i < numCameras; ++i) {
                frameCollectors.emplace_back(frameSize);
                calibrators.emplace_back();
                camThreads.emplace_back();
            }

            auto callback = [this](int cameraId, const MultiCameraCalibration &that) {
                cv::FileStorage storage;

                storage.open(std::format("gridDataset{}.yaml", cameraId), cv::FileStorage::WRITE);

                if (storage.isOpened()) {
                    frameCollectors[cameraId].store(storage);
                    storage.release();
                }

                storage.open(std::format("config_{}.yaml", cameraId), cv::FileStorage::WRITE);

                if (storage.isOpened()) {
                    getRectificationData(cameraId).store(storage);
                    storage.release();
                }

                storage.open(std::format("config_0_{}.yaml", cameraId), cv::FileStorage::WRITE);

                if (storage.isOpened()) {
                    getRectificationData(0).store(storage);
                    storage.release();
                }

                this->onUpdateCallback(cameraId, *this);
            };
            multicamThreadHandler = std::make_shared<MultiCameraCalibration>(
                    numCameras,
                    frameSize,
                    callback,
                    gridPreferredSizeProvider,
                    frameCollectors[0]
            );

            for (int i = 0; i < numCameras; ++i) {
                cameraThread.emplace_back(new SingleCameraThread(i, frameSize, [i, this] (const SingleCameraThread &that) {
                    if (this->onUpdateCallback != nullptr) {
                        this->onUpdateCallback(i, *this);
                        multicamThreadHandler->setCalibrationData(i, that.getCalibrationData());
                        multicamThreadWait.notify_all();
                    }
                }, gridPreferredSizeProvider, frameCollectors[i], calibrators[i]));
            }

            pendingFrames.resize(numCameras);
        }

        void loadConfig();

        [[nodiscard]] CalibrationData getCalibrationData(int cameraId) const {
            return cameraThread[cameraId]->getCalibrationData();
        }

        [[nodiscard]] CalibrationData getRectificationData(int cameraId) const {
            return multicamThreadHandler->getRectificationData(cameraId);
        }

        [[nodiscard]] cv::Point2d getF(int cameraId) {
            return cameraThread[cameraId]->getF();
        }

        [[nodiscard]] cv::Point2d getC(int cameraId) {
            return cameraThread[cameraId]->getC();
        }

        [[nodiscard]] cv::Mat getMap(int cameraId) const {
            return cameraThread[cameraId]->getMap();
        }

        [[nodiscard]] cv::Mat getRectifiedMap(int cameraId) const {
            return multicamThreadHandler->getRectifiedMap(cameraId);
        }

        [[nodiscard]] double getProgress(int cameraId) const {
            return cameraThread[cameraId]->getProgress();
        }

        [[nodiscard]] auto getGridSize() const {
            return gridPreferredSizeProvider.getGridPreferredSize();
        }

        [[nodiscard]] size_t getFrameCount(int cameraId) const {
            return cameraThread[cameraId]->getFrameCount();
        }

        [[nodiscard]] size_t getFrameSetCount(int cameraId) const {
            return cameraThread[cameraId]->getFrameSetCount();
        }

        [[nodiscard]] double getViewCosts(int cameraId) const {
            return cameraThread[cameraId]->getViewCosts();
        }

        [[nodiscard]] double getViewMulticamCosts(int cameraId) const {
            return multicamThreadHandler->getViewMulticamCosts(cameraId);
        }

        CalibrateFrameCollector::FrameRef
        createFrame(int cameraId, const std::vector<cv::Point3d> &imagePoints,
                    const std::vector<cv::Point3d> &objectPoints,
                    int w, int h, double cost, long ts);

        void addFrameSet(const FrameRefList &frameSet);

        void runCalibration();

        void stopCalibration() noexcept;

        ~CalibrationStrategy() {
            stopCalibration();
        }

        static void converPoints(const cv::Mat &pp, std::vector<ecv::CalibrateMapper::Point3> &points);
        static void converPoints(const std::vector<ecv::CalibrateMapper::Point3> &points, cv::Mat &pp);

        void undistortImagePoints(const std::vector<cv::Mat> &imagePoints, std::vector<cv::Mat> &plainPoints,
                                  const CalibrationData &calibrationData) const;

        void undistortImagePoints(const std::vector<ecv::CalibrateMapper::Point3> &imagePoints,
                                  std::vector<ecv::CalibrateMapper::Point3> &plainPoints,
                                  const CalibrationData &calibrationData) const;

        void rectifyImagePoints(const std::vector<cv::Mat> &imagePoints, std::vector<cv::Mat> &plainPoints,
                                const CalibrationData &calibrationData) const;

        void rectifyImagePoints(const std::vector<ecv::CalibrateMapper::Point3> &imagePoints,
                                std::vector<ecv::CalibrateMapper::Point3> &plainPoints,
                                const CalibrationData &calibrationData) const;
    };
}
#endif //EMBEDDED_CV_CALIBRATIONSTRATEGY_H
