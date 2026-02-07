#ifndef EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H
#define EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H

#include <vector>
#include <core/types.hpp>
#include <unordered_map>
#include <set>

namespace ecv {
    class CalibrateFrameCollector {
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
            const std::vector<cv::Point3d> imageGrid;
            const std::vector<cv::Point3d> objectGrid;
            size_t w;
            size_t h;
            double cost;
        };


        struct FrameCompare {
            bool operator()(const std::shared_ptr<Frame> &a, const std::shared_ptr<Frame> &b) const {
                if (a->cost == b->cost) return true;
                return a->cost > b->cost;
            }
        };

        const double RECT_MATCH_THRESHOLD = 0.7;
        const double NEAR_THRESHOLD = 0.5;
        const double FAR_THRESHOLD = 0.25;
        const double SMALL_ANGLE = 10;
        const double LARGE_ANGLE = 25;
        const size_t FRAMES_PER_CLASS = 5;

        const cv::Size &frameSize;

        std::unordered_map<FrameClass, std::multiset<std::shared_ptr<Frame>, FrameCompare>> map;
        std::unordered_map<std::shared_ptr<Frame>, int> frames;
        int frameId = 1;

        std::vector<FrameClass> getClasses(const Frame &frame) const;
    public:
        explicit CalibrateFrameCollector(const cv::Size &frameSize) : frameSize(frameSize) {};

        void addFrame(const std::vector<cv::Point3d> &imageGrid, const std::vector<cv::Point3d> &objectGrid, size_t w, size_t h, double cost);

        double getProgress() const;

        std::vector<std::vector<cv::Point3d>> getCollectedImageGrids() const;

        std::vector<std::vector<cv::Point3d>> getCollectedObjectGrids() const;

        auto getFrames() const {
            return frames;
        }
    };
};

#endif //EMBEDDED_CV_CALIBRATEFRAMECOLLECTOR_H
