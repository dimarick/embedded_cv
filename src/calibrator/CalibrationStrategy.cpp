#include "CalibrationStrategy.h"

using namespace ecv;

void CalibrationStrategy::runCalibration() {
    running = true;

    for (int i = 0; i < camThreads.size(); i++) {
        const auto that = this;
        camThreads[i] = std::thread([](decltype(this) that, int i) {
            std::vector<CalibrateFrameCollector::FrameRef> pendingFrames(MAX_FRAMES_QUEUE);
            while (that->running) {
                {
                    std::unique_lock<std::mutex> lock(that->pendingFramesMutex);
                    that->camThreadsWait.wait(lock);
                }
                if (that->running && !that->pendingFrames[i].empty()) {
                    {
                        std::unique_lock<std::mutex> lock2(that->pendingFramesMutex);
                        std::copy(that->pendingFrames[i].begin(), that->pendingFrames[i].end(), pendingFrames.begin());
                        pendingFrames.resize(that->pendingFrames[i].size());
                        that->pendingFrames[i].clear();
                    }
                    that->camThreadCallback(pendingFrames, i);
                }
            }
        }, this, i);
    }

    const auto that = this;
    multicamThread = std::thread([](decltype(this) that) {
        std::vector<std::vector<CalibrateFrameCollector::FrameRef>> frames;
        while (that->running) {
            {
                std::unique_lock<std::mutex> lock(that->pendingFramesMutex);
                that->multicamThreadWait.wait(lock);
            }
            if (that->running && !that->pendingFrameSets.empty()) {
                {
                    std::unique_lock<std::mutex> lock2(that->pendingFramesMutex);
                    frames.resize(that->pendingFrameSets.size());
                    std::copy(that->pendingFrameSets.begin(), that->pendingFrameSets.end(), frames.begin());
                    that->pendingFrameSets.clear();
                }
                that->multicamThreadCallback(frames);
            }
        }
    }, this);
}

void CalibrationStrategy::stopCalibration() noexcept {
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
                                 double ts) {
    return frameCollectors[cameraId].createFrame(imagePoints, objectPoints, w, h, cost, ts);
}

void CalibrationStrategy::addFrameSet(const std::vector<CalibrateFrameCollector::FrameRef> &frameSet) {
    CV_Assert(frameSet.size() == numCameras);
    {
        std::unique_lock<std::mutex> lock(pendingFramesMutex);
        pendingFrameSets.insert(frameSet);
        if (pendingFrameSets.size() > MAX_FRAMES_QUEUE) {
            pendingFrameSets.erase(pendingFrameSets.begin());
        }
        for (int i = 0; i < numCameras; ++i) {
            pendingFrames[i].insert(frameSet[i]);
            if (pendingFrames[i].size() > MAX_FRAMES_QUEUE) {
                pendingFrames[i].erase(pendingFrames[i].begin());
            }
        }
    }
    camThreadsWait.notify_all();
    multicamThreadWait.notify_all();
}

void CalibrationStrategy::camThreadCallback(const std::vector<CalibrateFrameCollector::FrameRef> &frames, int cameraId) {
    int i = cameraId;

    auto sample = frameCollectors[i].getFramesSample(SAMPLE_SIZE, false);

    for (const auto &frame : frames) {
        if (frame != nullptr) {
            frameCollectors[i].addFrame(frame);
            sample.emplace_back(frame);
        }
    }

    if (sample.empty()) {
        return;
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

    auto sampleValidate = frameCollectors[i].getFramesSample(SAMPLE_SIZE, true);

    if (sampleValidate.empty()) {
        return;
    }

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

void CalibrationStrategy::multicamThreadCallback(const std::vector<std::vector<CalibrateFrameCollector::FrameRef>> &frameSets) {
    std::vector<std::vector<CalibrateFrameCollector::FrameRef>> validFrameSets;

    for (const auto &frameSet : frameSets) {
        if (isValid(frameSet)) {
            validFrameSets.emplace_back(frameSet);
        }
    }

    if (validFrameSets.empty()) {
        return;
    }

    frameCollectors[0].addMulticamFrames(validFrameSets);

    auto mask = cv::Mat(numCameras, (int)validFrameSets.size(), CV_8U);
    std::vector<cv::Size> imageSize;
    std::vector<unsigned char> models;
    std::vector<int> flagsForIntrinsics;
    std::vector<cv::Mat> Rs;
    std::vector<cv::Mat> Ts;
    std::vector<cv::Mat> Ks;
    std::vector<cv::Mat> Ks2;
    std::vector<cv::Mat> Rs2;
    std::vector<cv::Mat> Ts2;
    std::vector<cv::Mat> distortions;
    std::vector<cv::Mat> distortions2;
    std::vector<cv::Mat> rvecs0(validFrameSets.size());
    std::vector<cv::Mat> tvecs0(validFrameSets.size());
    cv::Mat perFrameErrors;

    cv::Mat initializationPairs;
    auto objectPoints = getObjectPointsFromFrameSets(validFrameSets);
    auto imagePoints = getImagePointsFromFrameSets(validFrameSets);

    for (int frameId = 0; frameId < validFrameSets.size(); frameId++) {
        for (int camId = 0; camId < validFrameSets[frameId].size(); camId++) {
            mask.at<unsigned char >(camId, frameId) = validFrameSets[frameId][camId] == nullptr ? 0 : 255;
        }
    }

    for (int i = 0; i < numCameras; ++i) {
        const auto &calibrationData = getCalibrationData(i);

        imageSize.emplace_back(frameSize);
        models.emplace_back(cv::CALIB_MODEL_PINHOLE);
        flagsForIntrinsics.emplace_back(0);
        Ks.emplace_back(calibrationData.cameraMatrix);
        distortions.emplace_back(calibrationData.distCoeff);
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
            cv::CALIB_USE_INTRINSIC_GUESS,
            cv::TermCriteria(10, 1e-7)
    );

    const auto &sampleValidate = frameCollectors[0].getFrameSetsSample(VALIDATE_SAMPLE_SIZE, true);

    objectPoints = getObjectPointsFromFrameSets(sampleValidate);
    imagePoints = getImagePointsFromFrameSets(sampleValidate);

    if (sampleValidate.empty()) {
        return;
    }

    auto mask2 = cv::Mat(numCameras, (int)sampleValidate.size(), CV_8U);
    for (int frameId = 0; frameId < sampleValidate.size(); frameId++) {
        for (int camId = 0; camId < sampleValidate[frameId].size(); camId++) {
            mask2.at<unsigned char >(camId, frameId) = sampleValidate[frameId][camId] == nullptr ? 0 : 255;
        }
    }

    Ks2 = Ks;
    Rs2 = Rs;
    Ts2 = Ts;
    distortions2 = distortions;

    double cost = 1. / 0.;
    try {
        cost = cv::calibrateMultiview(
                objectPoints,
                imagePoints,
                imageSize,
                mask2,
                models,
                Ks2,
                distortions2,
                Rs2,
                Ts2,
                initializationPairs,
                rvecs0,
                tvecs0,
                perFrameErrors,
                flagsForIntrinsics,
                cv::CALIB_USE_INTRINSIC_GUESS,
                cv::TermCriteria(1, 1e-7)
        );
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    if (cost < multicamCosts) {
        multicamCosts = cost;

        for (int i = 0; i < numCameras; ++i) {
            cv::Mat tmp;
            auto calibrationData = getCalibrationData(i);
            calibrationData.cameraMatrix = Ks[i].clone();
            calibrationData.distCoeff = distortions[i];
            setCalibrationData(i, calibrationData);

//            cv::initUndistortRectifyMap(calibrationData.cameraMatrix, calibrationData.distCoeff, Rs[i], Ts[i], frameSize, CV_32FC2, map[i], tmp);
        }

        for (int i = 0; i < numCameras; ++i) {
            onUpdateCallback(i, *this);
        }
    }
}

std::vector<cv::Mat> CalibrationStrategy::getObjectPointsFromFrameSets(const std::vector<std::vector<CalibrateFrameCollector::FrameRef>> &frameSets) const {
    std::vector<cv::Mat> objectPoints;
    for (const auto & frameSet : frameSets) {
        std::vector<cv::Point3f> fp32Grid;
        for (const auto &p : frameSet[0]->objectGrid) {
            fp32Grid.emplace_back(p.x, p.y, 1.0f);
        }
        objectPoints.emplace_back(fp32Grid, CV_32FC3);
    }

    return objectPoints;
}

bool CalibrationStrategy::isValid(const std::vector<CalibrateFrameCollector::FrameRef> &frameSet) const {
    const auto &frame0 = frameSet[0];
    if (frame0 == nullptr) {
        return false;
    }
    size_t w = frame0->w, h = frame0->h;

    for (int camId = 1; camId < numCameras; camId++) {
        const auto &frame = frameSet[camId];
        if (frame == nullptr || frame->w != w || frame->h != h) {
            return false;
        }
    }

    return true;
}

std::vector<std::vector<cv::Mat>> CalibrationStrategy::getImagePointsFromFrameSets(const std::vector<std::vector<CalibrateFrameCollector::FrameRef>> &frameSets) const {
    std::vector<std::vector<cv::Mat>> imagePoints;
    for (int camId = 0; camId < numCameras; camId++) {
        imagePoints.emplace_back(frameSets.size());
    }

    for (int frameId = 0; frameId < frameSets.size(); frameId++) {
//        Expected shape: NUM_CAMERAS x NUM_FRAMES x NUM_POINTS x 2.
        const auto &frameSet = frameSets[frameId];
        for (int camId = 0; camId < numCameras; camId++) {
            std::vector<cv::Point2f> frameImagePoints;
            const auto &frame = frameSet[camId];
            for (const auto &p: frame->imageGrid) {
                frameImagePoints.emplace_back(p.x, p.y);
            }
            imagePoints[camId][frameId] = cv::Mat(frameImagePoints, CV_32FC2);
        }
    }

    return imagePoints;
}
