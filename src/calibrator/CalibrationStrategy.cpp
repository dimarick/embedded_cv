#include "CalibrationStrategy.h"
#include "CalibrateMapper.h"

using namespace ecv;

void CalibrationStrategy::runCalibration() {
    running = true;

    for (int i = 0; i < camThreads.size(); i++) {
        camThreads[i] = std::thread([](decltype(this) that, int i) {
            FrameRefList pendingFrames(MAX_FRAMES_QUEUE);
            while (that->running) {
                {
                    std::unique_lock lock(that->pendingFramesMutex);
                    that->camThreadsWait.wait(lock);
                }
                if (that->running && !that->pendingFrames[i].empty()) {
                    {
                        std::unique_lock lock2(that->pendingFramesMutex);
                        std::copy(that->pendingFrames[i].begin(), that->pendingFrames[i].end(), pendingFrames.begin());
                        pendingFrames.resize(that->pendingFrames[i].size());
                    }
                    that->cameraThread[i]->camThreadCallback(pendingFrames);
                    {
                        std::unique_lock lock2(that->pendingFramesMutex);
                        that->pendingFrames[i].clear();
                    }
                }
            }
        }, this, i);
    }

    multicamThread = std::thread([](decltype(this) that) {
        std::vector<FrameRefList> frames;
        while (that->running) {
            {
                std::unique_lock lock(that->pendingFramesMutex);
                that->multicamThreadWait.wait(lock);
            }
            if (that->running) {
                {
                    std::unique_lock lock2(that->pendingFramesMutex);
                    frames.resize(that->pendingFrameSets.size());
                    std::copy(that->pendingFrameSets.begin(), that->pendingFrameSets.end(), frames.begin());
                }
                that->multicamThreadHandler->multicamThreadCallback(frames);
                {
                    std::unique_lock lock2(that->pendingFramesMutex);
                    that->pendingFrameSets.clear();
                }
            }
        }
    }, this);
}

void CalibrationStrategy::stopCalibration() noexcept {
    if (!running) {
        return;
    }
    running = false;

    camThreadsWait.notify_all();
    multicamThreadWait.notify_all();

    for (auto &thread : camThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    if (multicamThread.joinable()) {
        multicamThread.join();
    }
}

CalibrateFrameCollector::FrameRef CalibrationStrategy::createFrame(int cameraId, const std::vector<cv::Point3d> &imagePoints,
                                 const std::vector<cv::Point3d> &objectPoints, int w, int h, double cost,
                                 long ts) {
    return frameCollectors[cameraId].createFrame(imagePoints, objectPoints, w, h, cost, ts);
}

void CalibrationStrategy::addFrameSet(const FrameRefList &frameSet) {
    if (frameSet[0] == nullptr) {
        return;
    }

    CV_Assert(frameSet.size() == numCameras);
    {
        std::unique_lock<std::mutex> lock(pendingFramesMutex);
        pendingFrameSets.insert(frameSet);
        if (pendingFrameSets.size() > MAX_FRAMES_QUEUE) {
            pendingFrameSets.erase(pendingFrameSets.begin());
        }
    }
    {
        std::unique_lock<std::mutex> lock(pendingFramesMutex);
        for (int i = 0; i < numCameras; ++i) {
            if (frameSet[i] != nullptr) {
                pendingFrames[i].insert(frameSet[i]);
                if (pendingFrames[i].size() > MAX_FRAMES_QUEUE) {
                    pendingFrames[i].erase(pendingFrames[i].begin());
                }
            }
        }
    }
    camThreadsWait.notify_all();
    multicamThreadWait.notify_all();
}

void CalibrationStrategy::loadConfig() {
    for (int i = 0; i < numCameras; ++i) {
        cv::FileStorage storage;
        storage.open(std::format("gridDataset{}.yaml", i), cv::FileStorage::READ);

        if (!storage.isOpened()) {
            continue;
        }

        frameCollectors[i].load(gridPreferredSizeProvider, storage);

        storage.release();
    }

    auto [w, h] = gridPreferredSizeProvider.getGridPreferredSize();

    if (w == 0) {
        return;
    }

    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < numCameras; ++i) {
            const auto &framesSample = frameCollectors[i].getFramesSample(20, w, h, false);
            cameraThread[i]->camThreadCallback(framesSample);
        }

        const auto &framesSample = frameCollectors[0].getFrameSetsSample(20, w, h, false);
        multicamThreadHandler->multicamThreadCallback(framesSample);
    }
}

void CalibrationStrategy::undistortImagePoints(const std::vector<cv::Mat> &imagePoints, std::vector<cv::Mat> &plainPoints, const CalibrationData &calibrationData) const {
    SingleCameraThread::undistortImagePoints(imagePoints, plainPoints, calibrationData);
}

void CalibrationStrategy::undistortImagePoints(const std::vector<ecv::CalibrateMapper::Point3> &imagePoints, std::vector<ecv::CalibrateMapper::Point3> &plainPoints, const CalibrationData &calibrationData) const {
    SingleCameraThread::undistortImagePoints(imagePoints, plainPoints, calibrationData);
}

void CalibrationStrategy::rectifyImagePoints(const std::vector<cv::Mat> &imagePoints, std::vector<cv::Mat> &plainPoints, const CalibrationData &calibrationData) const {
    MultiCameraCalibration::rectifyImagePoints(imagePoints, plainPoints, calibrationData);
}


void CalibrationStrategy::rectifyImagePoints(const std::vector<ecv::CalibrateMapper::Point3> &imagePoints, std::vector<ecv::CalibrateMapper::Point3> &plainPoints, const CalibrationData &calibrationData) const {
    MultiCameraCalibration::rectifyImagePoints(imagePoints, plainPoints, calibrationData);
}

void CalibrationStrategy::converPoints(const cv::Mat &pp, std::vector<ecv::CalibrateMapper::Point3> &points) {
    for (int i = 0; i < pp.total(); ++i) {
        const auto &p = pp.at<cv::Point2f>(i);
        points[i].x = p.x;
        points[i].y = p.y;
        points[i].z = 0.0;
    }
}


void CalibrationStrategy::converPoints(const std::vector<ecv::CalibrateMapper::Point3> &points, cv::Mat &pp) {
    pp = cv::Mat::zeros((int)points.size(), 1, CV_32FC2);
    for (int i = 0; i < points.size(); ++i) {
        pp.at<cv::Point2f>(i) = cv::Point2f((float)points[i].x, (float)points[i].y);
    }
}

