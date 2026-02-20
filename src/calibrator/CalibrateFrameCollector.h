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
            int cls;
            const std::vector<cv::Point3d> imageGrid;
            const std::vector<cv::Point3d> objectGrid;
            size_t w;
            size_t h;
            double cost;
            double ts;
            bool validate;
        };
        typedef std::shared_ptr<Frame> FrameRef;
    private:
        cv::Size frameSize;

        const int NUM_CLASSES = 30*30*30;
        const int CLASSES_CUBE_DIM = 3;
        const int CLASSES_CUBE_SIZE = (int)std::round(std::pow(NUM_CLASSES, 1. / (double)CLASSES_CUBE_DIM));

        std::unordered_map<int, FrameRef> map;
        std::unordered_map<int, std::vector<FrameRef>> frameSets;

        double maxRotValue[2] = {0,0};
        double maxDistValue = 0;
        double minDistValue = 1;
        void addFrameTo(GridPreferredSizeProvider &gridPreferredSizeProvider, decltype(map) *m, const FrameRef &frameRef);
    public:
        int getClass(const Frame &frame);
        FrameRef createFrame(const std::vector<cv::Point3d> &imageGrid, const std::vector<cv::Point3d> &objectGrid, size_t w, size_t h, double cost, double ts);
        explicit CalibrateFrameCollector(cv::Size frameSize) : frameSize(frameSize) {}
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
        int getClassesCubeSize() const { return CLASSES_CUBE_SIZE; }
    };
};

#endif //EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H
