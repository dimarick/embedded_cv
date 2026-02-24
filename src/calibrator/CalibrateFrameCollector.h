#ifndef EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H
#define EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H

#include <memory>
#include <vector>
#include "opencv2/core.hpp"
#include "GridPreferredSizeProvider.h"
#include <unordered_map>
#include <set>
#include <mutex>

namespace ecv {
    class CalibrateFrameCollector {
    public:
        struct Frame {
            // тангенсы углов поворота по X, Y и Z, от 0 (наклона нет) до 1 (сильный наклон)
            cv::Point3d rotation;
            // xy - позиция верхнего левого угла габаритного прямоугольника сетки,
            // измеряемая в долях от размера кадра минус размер прямоугольника, от 0 до 1
            // z - корень из отношения площадей кадра и этого прямоугольника, от 0 до 1
            cv::Point3d position;
            // класс для хеш-таблицы, x * D1 + y * D1 * D2 + z * D1 * D2 * D3, где D1-3 - размерности куба параметров
            int rotationClass;
            int positionClass;
            const std::vector<cv::Point3d> imageGrid;
            const std::vector<cv::Point3d> objectGrid;
            size_t w;
            size_t h;
            double cost;
            double ts;
            bool validate;
        };
        typedef std::shared_ptr<const Frame> FrameRef;
    private:
        mutable std::unique_ptr<std::mutex> m;
        cv::Size frameSize;

        struct Dim {
            int size;
            double bias;
            double scale;
        };

        std::unordered_map<int, FrameRef> rotationMap;
        std::unordered_map<int, FrameRef> positionMap;
        std::unordered_map<int, std::vector<FrameRef>> frameSets;

        void addFrameTo(GridPreferredSizeProvider &gridPreferredSizeProvider, int cls, std::unordered_map<int, FrameRef> *m, const FrameRef &frameRef);
    public:
        explicit CalibrateFrameCollector(cv::Size frameSize) : frameSize(frameSize) {
            m = std::make_unique<std::mutex>();
        }
        const Dim R_DIM_X = {10, 0.5, 0.5};
        const Dim R_DIM_Y = {10, 0.5, 0.5};
        const Dim R_DIM_Z = {10, 0.5, 0.5};

        const Dim P_DIM_X = {10, 0.0, 1.0};
        const Dim P_DIM_Y = {10, 0.0, 1.0};
        const Dim P_DIM_Z = {10, 0.0, 1.0};

        const int TOTAL_VOLUME = R_DIM_X.size * R_DIM_Y.size * R_DIM_Z.size + P_DIM_X.size * P_DIM_Y.size * P_DIM_Z.size;

        cv::Point3d getRotationClass(const CalibrateFrameCollector::Frame &frame) const;
        cv::Point3d getPositionClass(const Frame &frame) const;
        int getClass(cv::Point3d point3, Dim dimX, Dim dimY, Dim dimZ);
        FrameRef createFrame(const std::vector<cv::Point3d> &imageGrid, const std::vector<cv::Point3d> &objectGrid, size_t w, size_t h, double cost, double ts, bool validate = std::rand() % 2 == 0);
        void addFrame(GridPreferredSizeProvider &gridPreferredSizeProvider, const FrameRef &frameRef);
        void addMulticamFrames(const std::vector<std::vector<FrameRef>>& _frameSets);
        double getProgress() const;
        std::vector<FrameRef> getFramesSample(int n, size_t w, size_t h, bool validate) const;
        std::vector<std::vector<FrameRef>> getFrameSetsSample(int n, size_t w, size_t h, bool validate) const;
        std::vector<std::vector<cv::Point3d>> getCollectedImageGridsSample(const std::vector<FrameRef> &sample) const;
        std::vector<std::vector<cv::Point3d>> getCollectedObjectGridsSample(const std::vector<FrameRef> &sample) const;
        FrameRef loadFrame(const cv::FileNode &frame);
        void load(GridPreferredSizeProvider &gridPreferredSizeProvider, const cv::FileStorage &fs);
        void store(cv::FileStorage &fs) const;
        int getDatasetVolume() const { return TOTAL_VOLUME; }

        std::vector<CalibrateFrameCollector::FrameRef>
        getFramesSampleFrom(int n, size_t w, size_t h, bool validate,
                            const std::unordered_map<int, FrameRef> &map) const;

        size_t getFrameCount() const;
        size_t getFrameSetCount() const;
    };
};

#endif //EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H
