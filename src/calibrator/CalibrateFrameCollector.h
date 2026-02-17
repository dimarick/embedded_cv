#ifndef EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H
#define EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H

#include <vector>
#include "opencv2/core.hpp"
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

        struct FramePair {
            double cost;
            FrameRef base;
            FrameRef current;

            bool isValidate() {
                return base->validate;
            }

            int getClass() {
                return base->cls;
            }
        };
        typedef std::shared_ptr<FramePair> FramePairRef;
    private:
        const cv::Size &frameSize;

        const int NUM_CLASSES = 30*30*30;
        const int CLASSES_CUBE_DIM = 3;
        const int CLASSES_CUBE_SIZE = (int)std::round(std::pow(NUM_CLASSES, 1. / (double)CLASSES_CUBE_DIM));

        std::unordered_map<int, FrameRef> map;
        std::unordered_map<int, FramePairRef> pairs;
        std::unordered_map<int, std::vector<FrameRef>> frameSets;

        double maxRotValue[2] = {0,0};
        double maxDistValue = 0;
        double minDistValue = 1;

        int getClass(const Frame &frame);
        void addFrameTo(decltype(map) *m, const FrameRef &frameRef);
        void addMulticamFrameTo(decltype(pairs) *m, const CalibrateFrameCollector::FramePairRef &framePairRef);
    public:
        FrameRef createFrame(const std::vector<cv::Point3d> &imageGrid, const std::vector<cv::Point3d> &objectGrid, size_t w, size_t h, double cost, double ts);
        explicit CalibrateFrameCollector(const cv::Size &frameSize) : frameSize(frameSize) {};
        void addFrame(const FrameRef &frameRef);
        void addMulticamFrame(const FrameRef &baseFrameRef, const FrameRef &frameRef, double cost);
        void addMulticamFrames(const std::vector<std::vector<CalibrateFrameCollector::FrameRef>>& _frameSets);

        double getProgress() const;
        std::vector<std::pair<int, FrameRef>> getFramesSample(int n) const;
        std::vector<std::pair<int, CalibrateFrameCollector::FramePairRef>> getFramesPairsSample(int n) const;
        std::vector<std::vector<FrameRef>> getFramesSample2(int n, bool validate) const;
        std::vector<std::vector<cv::Point3d>> getCollectedImageGridsSample(const std::vector<FrameRef> &sample) const;
        std::vector<std::vector<cv::Point3d>> getCollectedObjectGridsSample(const std::vector<FrameRef> &sample) const;
        template <typename Iterator> std::vector<std::vector<cv::Point3d>> getCollectedImageGridsSample(Iterator first, Iterator last) const {
            std::vector<std::vector<cv::Point3d>> result;

            for (auto &it = first; it != last; ++it) {
                result.emplace_back((*it)->imageGrid);
            }

            return result;
        }
        template <typename Iterator> std::vector<std::vector<cv::Point3d>> getCollectedObjectGridsSample(Iterator first, Iterator last) const {
            std::vector<std::vector<cv::Point3d>> result;

            for (auto &it = first; it != last; ++it) {
                result.emplace_back((*it)->objectGrid);
            }

            return result;
        }

        FrameRef loadFrame(const cv::FileNode &frame);
        void load(const cv::FileStorage &fs);
        void store(cv::FileStorage &fs) const;

    };
};

#endif //EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H
