#include "CalibrationStrategy.h"

using namespace ecv;

void CalibrationStrategy::runCalibration() {
    running = true;

    for (int i = 0; i < camThreads.size(); i++) {
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

    auto trainData = getCalibrationData(i);

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
        setCalibrationData(i, trainData);
        costs[i] = cost;

        std::cout << "Calib " << cameraId << ", c = " << cost << std::endl;
    }
}

void CalibrationStrategy::multicamThreadCallback(const std::vector<std::vector<CalibrateFrameCollector::FrameRef>> &frameSets) {
    std::vector<std::vector<CalibrateFrameCollector::FrameRef>> validFrameSets;

    for (int i = 0; i < numCameras; ++i) {
        if (costs[i] > 10) {
            return;
        }
    }

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
    std::vector<cv::Mat> distortions;
    std::vector<cv::Mat> rvecs0;
    std::vector<cv::Mat> tvecs0;
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
        Rs.emplace_back();
        Ts.emplace_back();
        distortions.emplace_back(calibrationData.distCoeff, CV_64FC1);
    }

    double trainCost = 1. / 0.;
    try {
        trainCost = cv::calibrateMultiview(
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
                cv::TermCriteria(1000, 1e-7)
        );
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }


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

    std::vector<cv::Mat> Ks2(numCameras);
    std::vector<cv::Mat> Rs2(numCameras);
    std::vector<cv::Mat> Ts2(numCameras);
    std::vector<cv::Mat> distortions2(numCameras);
    std::vector<cv::Mat> rvecs2;
    std::vector<cv::Mat> tvecs2;
    cv::Mat initializationPairs2;

    for (int i = 0; i < numCameras; ++i) {
        Ks2[i] = Ks[i].clone();
        distortions2[i] = distortions[i].clone();
    }

    double cost = trainCost;//1. / 0.;
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
                initializationPairs2,
                rvecs2,
                tvecs2,
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
            calibrationData.cameraMatrix = Ks[i];
            calibrationData.distCoeff = distortions[i];
            setCalibrationData(i, calibrationData);
            cv::Mat R;

            cv::Rodrigues(Rs[i], R);

            cv::initUndistortRectifyMap(calibrationData.cameraMatrix, calibrationData.distCoeff, R, calibrationData.cameraMatrix, frameSize, CV_32FC2, map[i], tmp);
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
            fp32Grid.emplace_back(p.x, p.y, 0.0f);
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
