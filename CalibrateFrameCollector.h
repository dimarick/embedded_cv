#ifndef EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H
#define EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H

#include <vector>
#include <core/types.hpp>
#include <unordered_map>
#include <set>
#include <mutex>

namespace ecv {
    class CalibrateFrameCollector {
    public:
        enum FrameClass {
            topLeft = 1,
            topRight,
            bottomLeft,
            bottomRight,
            center,
            near,
            mid,
            far,
            rollSmall,
            yawSmall,
            pitchSmall,
            rollMedium,
            yawMedium,
            pitchMedium,
            rollLarge,
            yawLarge,
            pitchLarge,
            rollSmallN,
            yawSmallN,
            pitchSmallN,
            rollMediumN,
            yawMediumN,
            pitchMediumN,
            rollLargeN,
            yawLargeN,
            pitchLargeN,
            COUNT,
        };

        struct Frame {
            std::vector<FrameClass> classes;
            const std::vector<cv::Point3d> imageGrid;
            const std::vector<cv::Point3d> objectGrid;
            size_t w;
            size_t h;
            double cost;
        };

        struct FramePair {
            double cost;
            std::shared_ptr<Frame> base;
            std::shared_ptr<Frame> current;
        };

        const double RECT_MATCH_THRESHOLD = 0.7;
        const double NEAR_THRESHOLD = 0.5;
        const double FAR_THRESHOLD = 0.25;
        const double SMALL_ANGLE = 10;
        const double LARGE_ANGLE = 25;
        const size_t FRAMES_PER_CLASS = 3;
    private:
        struct FrameCompare {
            bool operator()(const std::shared_ptr<Frame> &a, const std::shared_ptr<Frame> &b) const {
                if (a->cost == b->cost) return true;
                return a->cost > b->cost;
            }
        };

        struct FramePairCompare {
            bool operator()(const std::shared_ptr<FramePair> &a, const std::shared_ptr<FramePair> &b) const {
                if (a->cost == b->cost) return true;
                return a->cost > b->cost;
            }
        };

        const cv::Size &frameSize;

        std::unordered_map<FrameClass, std::multiset<std::shared_ptr<Frame>, FrameCompare>> map;
        std::unordered_map<FrameClass, std::multiset<std::shared_ptr<FramePair>, FramePairCompare>> pairs;
        std::unordered_map<std::shared_ptr<Frame>, int> frames;

        // frame pairs indexed by long frameSetId and int cameraId
//        std::mutex framePairsMutex;
        std::unordered_map<long, FramePair> framePairs;

        std::vector<FrameClass> getClasses(const Frame &frame) const;
    public:
        explicit CalibrateFrameCollector(const cv::Size &frameSize) : frameSize(frameSize) {};

        void addFrame(const std::shared_ptr<Frame> &frameRef);
        std::shared_ptr<Frame> createFrame(const std::vector<cv::Point3d> &imageGrid, const std::vector<cv::Point3d> &objectGrid, size_t w, size_t h, double cost);

        void addMulticamFrame(const std::shared_ptr<Frame> &baseFrameRef, const std::shared_ptr<Frame> &frameRef, double cost);

        double getProgress() const;

        std::set<std::shared_ptr<FramePair>> getValidFramePairs();

        std::vector<std::vector<cv::Point3d>> getCollectedImageGrids() const;

        std::vector<std::vector<cv::Point3d>> getCollectedObjectGrids() const;

        std::shared_ptr<Frame> loadFrame(const cv::FileNode &frame);
        void load(const cv::FileStorage &fs);
        void store(cv::FileStorage &fs) const;

        auto getFrames() const {
            return frames;
        }

    };
};

#endif //EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H
