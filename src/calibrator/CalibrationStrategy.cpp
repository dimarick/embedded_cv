#include "CalibrationStrategy.h"

using namespace ecv;

void CalibrationStrategy::runCalibration() {
    running = true;

    for (int i = 0; i < camThreads.size(); i++) {
        const auto that = this;
        camThreads[i] = std::thread([](decltype(this) that, int i) {
            std::vector<CalibrateFrameCollector::FrameRef> pendingFrames(MAX_FRAMES_QUEUE);
            while (that->running) {
                std::unique_lock<std::mutex> lock(that->pendingFramesMutex);
                that->camThreadsWait[i].wait(lock);
                if (that->running && !that->pendingFrames.empty()) {
                    {
                        std::unique_lock<std::mutex> lock2(that->pendingFramesMutex);
                        std::copy(that->pendingFrames[i].begin(), that->pendingFrames[i].end(), pendingFrames.begin());
                        that->pendingFrames.clear();
                    }
                    that->camThreadCallback(pendingFrames, i);
                }
            }
        }, this, i);
    }

    if (multicamThread.joinable()) {
        const auto that = this;
        multicamThread = std::thread([](decltype(this) that) {
            std::vector<std::set<CalibrateFrameCollector::FrameRef>> frames;
            while (that->running) {
                std::unique_lock<std::mutex> lock(that->pendingFramesMutex);
                that->multicamThreadWait.wait(lock);
                if (that->running && !that->pendingFrames2.empty()) {
                    {
                        std::unique_lock<std::mutex> lock2(that->pendingFramesMutex);
                        std::copy(that->pendingFrames2.begin(), that->pendingFrames2.end(), frames.begin());
                        that->pendingFrames2.clear();
                    }
                    that->multicamThreadCallback(frames);
                }
            }
        }, this);
    }
}

void CalibrationStrategy::stopCalibration() {
    running = false;
    for (auto &w : camThreadsWait) {
        w.notify_one();
    }

    multicamThreadWait.notify_one();

    for (auto &thread : camThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    if (multicamThread.joinable()) {
        multicamThread.join();
    }
}

void CalibrationStrategy::addFrame(int cameraId,
                                     const std::vector<cv::Point3d>& imagePoints,
                                     const std::vector<cv::Point3d>& objectPoints,
                                     int w, int h, double cost, double ts) {
    auto frameRef = frameCollectors[cameraId].createFrame(imagePoints, objectPoints, w, h, cost, ts);
    std::unique_lock<std::mutex> lock(pendingFramesMutex);
    pendingFrames[cameraId].insert(frameRef);
    if (pendingFrames[cameraId].size() > MAX_FRAMES_QUEUE) {
        pendingFrames[cameraId].erase(pendingFrames[cameraId].begin());
    }

    camThreadsWait[cameraId].notify_one();
}

void CalibrationStrategy::camThreadCallback(const std::vector<CalibrateFrameCollector::FrameRef> &frames, int cameraId) {
    int i = cameraId;

    for (const auto &frame : frames) {
        frameCollectors[cameraId].addFrame(frame);
    }

    std::vector<ecv::CalibrateFrameCollector::FrameRef> sample;
    std::vector<ecv::CalibrateFrameCollector::FrameRef> sampleValidate;

    auto sampleRaw = frameCollectors[i].getFramesSample(SAMPLE_SIZE);

    for (const auto &item : sampleRaw) {
        if (item.second->validate) {
            sampleValidate.emplace_back(item.second);
        } else {
            sample.emplace_back(item.second);
        }
    }

    auto trainData = data[i];

    calibrators[i].calibrateSingleCamera(
            frameSize,
            frameCollectors[i].getCollectedObjectGridsSample(sample),
            frameCollectors[i].getCollectedImageGridsSample(sample),
            trainData,
            0,
            cv::TermCriteria(10, 1e-7)
    );

    auto testData = trainData;

    int flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_TILTED_MODEL |
                cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_FIX_PRINCIPAL_POINT |
                cv::CALIB_FIX_FOCAL_LENGTH | cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 |
                cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6 | cv::CALIB_FIX_S1_S2_S3_S4 |
                cv::CALIB_FIX_TAUX_TAUY | cv::CALIB_FIX_TANGENT_DIST | cv::CALIB_FIX_INTRINSIC |
                cv::CALIB_FIX_SKEW;

    auto cost = calibrators[i].calibrateSingleCamera(
            frameSize,
            frameCollectors[i].getCollectedObjectGridsSample(sampleValidate),
            frameCollectors[i].getCollectedImageGridsSample(sampleValidate),
            testData,
            flags,
            cv::TermCriteria(1, 1e-7)
    );

    if (cost < costs[i]) {
        costs[i] = cost;
    }

    pendingFrames2[i].insert(frames.begin(), frames.end());

    onUpdateCallback(cameraId);

    multicamThreadWait.notify_one();
}

void CalibrationStrategy::multicamThreadCallback(const std::vector<std::set<CalibrateFrameCollector::FrameRef>> &framesPerCam) {

}
