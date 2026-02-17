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

CalibrateFrameCollector::FrameRef CalibrationStrategy::findClosestFrameByTs(
        const std::set<CalibrateFrameCollector::FrameRef, TsCompare> &set, double v) const {
    if (set.empty()) {
        return nullptr;
    }

    // бинарный поиск первого элемента с ts >= v
    auto compare = [](const CalibrateFrameCollector::FrameRef &a, double v) { return a->ts < v; };
    auto it = set.lower_bound(v);

    if (it == set.end()) {
        return *std::prev(it);
    }

    auto found = *it;

    if (found->ts == v || it == set.begin()) {
        return found;
    }

    const auto &prev = *std::prev(it);
    if (std::abs(v - found->ts) < std::abs(v - prev->ts)) {
        return found;
    }

    return prev;
}

/**
 * Из массива кадров находит пары (для стерео) или кортежи (для многокамерных систем) кадров, такие что:
 * - кадры близки по времени
 * - размеры сетки совпадают
 * - размер сетки не менее чем 4х4
 *
 * @param framesPerCam
 * @return
 */
std::vector<std::vector<CalibrateFrameCollector::FrameRef>> CalibrationStrategy::getFrameSets(const std::vector<std::set<CalibrateFrameCollector::FrameRef>> &framesPerCam) {
    std::vector<std::vector<CalibrateFrameCollector::FrameRef>> result;

    std::vector<std::set<CalibrateFrameCollector::FrameRef, TsCompare>> sortedFrames;
    std::vector<std::vector<CalibrateFrameCollector::FrameRef>> sortedFramesVector;

    for (const auto &camFrames : framesPerCam) {
        const auto &sorted = sortedFrames.emplace_back(camFrames.begin(), camFrames.end());
        sortedFramesVector.emplace_back(sorted.begin(), sorted.end());
    }

    auto interval = getFrameTimeInterval(sortedFrames);

    for (int baseCam = 0; baseCam < sortedFramesVector.size(); baseCam++) {
        result.emplace_back(framesPerCam.size());
        for (int f = 0; f < sortedFramesVector[baseCam].size(); ++f) {
            const auto &frame = sortedFramesVector[baseCam][f];
            for (int pairCam = 0; pairCam < sortedFramesVector.size(); ++pairCam) {
                if (baseCam == pairCam) {
                    continue;
                }

                if (result[f][pairCam] != nullptr) {
                    continue;
                }

                const auto &pair = findClosestFrameByTs(sortedFrames[pairCam], frame->ts);
                if (std::abs(pair->ts - sortedFramesVector[baseCam][f]->ts) > interval) {
                    continue;
                }
                if (frame->w < 4 || frame->h < 4 || frame->w != pair->w || frame->h != pair->h) {
                    continue;
                }

                result[f][baseCam] = frame;
                result[f][pairCam] = pair;
            }
        }
    }

    int j = 0;
    for (int i = 0; i < result.size(); i++) {
        auto valid = false;
        for (const auto &k : result[i]) {
            if (k == nullptr) {
                continue;
            }
            valid = true;
        }
        if (valid && j < i) {
            result[j] = result[i];
            j++;
        }
    }

    result.resize(j);

    return result;
}

double CalibrationStrategy::getFrameTimeInterval(const std::vector<std::set<CalibrateFrameCollector::FrameRef, TsCompare>> &framesPerCam) const {
    double frameMinInterval = 1. / 0.;

    for (const auto &camFrames : framesPerCam) {
        if (camFrames.size() < 2) {
            continue;
        }

        double interval = 1. / 0.;
        CalibrateFrameCollector::FrameRef prev = *camFrames.begin();
        for (const auto &frame : camFrames | std::views::drop(1)) {
            interval = std::min(interval, frame->ts - prev->ts);
            prev = frame;
        }

        frameMinInterval = std::min(frameMinInterval, interval);
    }

    return frameMinInterval;
}

void CalibrationStrategy::multicamThreadCallback(const std::vector<std::set<CalibrateFrameCollector::FrameRef>> &framesPerCam) {
    const auto &frameSets = getFrameSets(framesPerCam);

    frameCollectors[0].addMulticamFrames(frameSets);

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<std::vector<cv::Point2f>>> imagePoints;
    std::vector<std::vector<unsigned char>> mask;
    std::vector<cv::Size> imageSize;
    std::vector<int> models;
    std::vector<int> flagsForIntrinsics;
    std::vector<cv::Mat> Rs;
    std::vector<cv::Mat> Ts;
    std::vector<cv::Mat> Ks;
    std::vector<cv::Mat> Ks2;
    std::vector<cv::Mat> distortions;
    std::vector<cv::Mat> distortions2;
    std::vector<cv::Mat> rvecs0(frameSets.size());
    std::vector<cv::Mat> tvecs0(frameSets.size());
    std::vector<double> perFrameErrors(frameSets.size());
    cv::Mat initializationPairs;

    for (int i = 0; i < frameSets.size(); i++) {
        objectPoints.emplace_back(frameSets[i][0]->objectGrid.size());
        for (const auto &p : frameSets[i][0]->objectGrid) {
            objectPoints[i].emplace_back(p.x, p.y, 0);
        }
        imagePoints.emplace_back(frameSets[i][0]->imageGrid.size());
        for (int j = 0; j < frameSets[i].size(); j++) {
            if (frameSets[i][j] == nullptr) {
                mask[i][j] = 0;
                continue;
            }
            mask[i][j] = 255;
            for (const auto &p: frameSets[i][j]->imageGrid) {
                imagePoints[i][j].emplace_back(p.x, p.y);
            }
        }
    }

    for (int i = 0; i < numCameras; ++i) {
        imageSize.emplace_back(frameSize);
        models.emplace_back(cv::CALIB_MODEL_PINHOLE);
        flagsForIntrinsics.emplace_back(0);
        Ks.emplace_back(data[i].cameraMatrix.clone());
        distortions.emplace_back(data[i].distCoeff);
    }

    cv::calibrateMultiview(
            objectPoints,
            imagePoints,
            imageSize,
            mask,
            models,
            Ks,
            distortions,
            Rs,
            Ts,
            initializationPairs,
            rvecs0,
            tvecs0,
            perFrameErrors,
            flagsForIntrinsics,
            0,
            cv::TermCriteria(10, 1e-7)
    );

    const auto &sample = frameCollectors[0].getFramesSample2(100, true);

    objectPoints.clear();
    imagePoints.clear();

    for (int i = 0; i < frameSets.size(); i++) {
        objectPoints.emplace_back(frameSets[i][0]->objectGrid.size());
        for (const auto &p : frameSets[i][0]->objectGrid) {
            objectPoints[i].emplace_back(p.x, p.y, 0);
        }
        imagePoints.emplace_back(frameSets[i][0]->imageGrid.size());
        for (int j = 0; j < frameSets[i].size(); j++) {
            if (frameSets[i][j] == nullptr) {
                mask[i][j] = 0;
                continue;
            }
            mask[i][j] = 255;
            for (const auto &p: frameSets[i][j]->imageGrid) {
                imagePoints[i][j].emplace_back(p.x, p.y);
            }
        }
    }

    for (int i = 0; i < numCameras; ++i) {
        Ks2.emplace_back(data[i].cameraMatrix.clone());
        distortions2.emplace_back(data[i].distCoeff);
    }

    double cost = cv::calibrateMultiview(
            objectPoints,
            imagePoints,
            imageSize,
            mask,
            models,
            Ks,
            distortions,
            Rs,
            Ts,
            initializationPairs,
            rvecs0,
            tvecs0,
            perFrameErrors,
            flagsForIntrinsics,
            0,
            cv::TermCriteria(1, 1e-7)
    );

    if (cost < multicamCosts) {
        multicamCosts = cost;

        for (int i = 0; i < numCameras; ++i) {
            data[i].cameraMatrix = Ks[i].clone();
            data[i].distCoeff = distortions[i];
        }
    }
}
