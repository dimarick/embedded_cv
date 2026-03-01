#include <random>
#include "SingleCameraThread.h"

using namespace ecv;

void SingleCameraThread::camThreadCallback(const FrameRefList &frames) {
    size_t w, h;
    if (frames.empty()) {
        std::tie(w, h) = gridPreferredSizeProvider.getGridPreferredSize();
    } else {
        w = frames.begin()->get()->w;
        h = frames.begin()->get()->h;
    }

    auto sample = frameCollector.getFramesSample(TRAIN_SAMPLE_SIZE, w, h, false);

    for (const auto &frame : frames) {
        if (frame != nullptr && frameCollector.addFrame(gridPreferredSizeProvider, frame)) {
            if (!frame->validate) {
                sample.emplace_back(frame);
            }
        }
    }

    if (sample.empty()) {
        return;
    }

    auto trainData = tmpData;
    auto isFirstTime = trainData.frameCount == 0;

    std::vector<double> stdDeviationsIntrinsics(CALIB_NINTRINSIC);
    std::vector<double> perViewErrors(sample.size());

    calibrator.calibrateSingleCamera(
            frameSize,
            frameCollector.getCollectedObjectGridsSample(sample),
            frameCollector.getCollectedImageGridsSample(sample),
            trainData,
            stdDeviationsIntrinsics,
            perViewErrors,
            0,
            cv::TermCriteria(100, 1e-8)
    );

    if (!isFirstTime) {
        auto existsData = getCalibrationData();
        double ema = 2. / (1. + 1.);
        calibrator.mergeIntrinsics(trainData, ema, existsData, 0.05);
        trainData = existsData;
    }

    auto testData = trainData;
    std::tie(w, h) = gridPreferredSizeProvider.getGridPreferredSize();
    auto sampleValidate = frameCollector.getFramesSample(VALIDATE_SAMPLE_SIZE, w, h, true);

    if (sampleValidate.empty()) {
        return;
    }

    double gridCost = 0.0;
    double gridPlainCost = 0.0;

    for (const auto &frame : sampleValidate) {
        std::vector<ecv::CalibrateMapper::Point3> plainPoints(frame->imageGrid.size());
        undistortImagePoints(frame->imageGrid, plainPoints, trainData);
        gridPlainCost = std::max(gridPlainCost, (double)CalibrateMapper::getGridCost(plainPoints, (int)frame->w, (int)frame->h));
        gridCost += (double)CalibrateMapper::getGridCost(frame->imageGrid, (int)frame->w, (int)frame->h);
    }

    gridCost = gridPlainCost / gridCost;

    auto testCost = calibrator.validateSingleCamera(
            frameSize,
            frameCollector.getCollectedObjectGridsSample(sampleValidate),
            frameCollector.getCollectedImageGridsSample(sampleValidate),
            testData);

    std::uniform_real_distribution<double> randomRange(0., 1.);
    std::mt19937 r {std::random_device{}()};
    double annealEma = 2. / (20. + 1.);

    // Имитация отжига. Вероятность перехода тем выше, чем меньше ошибка и чем выше температура.
    // Отношение нормируется к 0.0-1.0 с помощью e^(d/T)
    double random = randomRange(r);
    auto probability = std::pow(std::numbers::e, (reprCost - testCost) / temperature);
    bool condition = testCost < reprCost * 1.1 && gridCost < gridMaxCost * 1.5;
    if (condition || random < probability) {
        std::cout << (condition ? "Single cam Normal... " : "Annealing... ") << cameraId << std::endl;
        tmpData = trainData;
        if (condition) {
            setCalibrationData(trainData);
            testCost = std::min(testCost, testCost);
            viewCost = testCost;
            reprCost = testCost;

            cv::Mat tmp;
            cv::initUndistortRectifyMap(trainData.cameraMatrix, trainData.distCoeff, cv::noArray(),
                                        trainData.cameraMatrix, frameSize, CV_32FC2, map, tmp);

            onUpdateCallback(*this);

            failCount = 0;
        }
    }

    if (failCount > 30) {
        temperature *= 1.5;
    } else {
        temperature *= 0.97;
    }

    if (temperature > 100) {
        reprCost = 1. / 0.;
        temperature = 10;
        tmpData = CalibrationData(frameSize);
        failCount = 0;
    }

    failCount++;
}


void SingleCameraThread::undistortImagePoints(const std::vector<cv::Mat> &imagePoints, std::vector<cv::Mat> &plainPoints, const CalibrationData &calibrationData) const {
    const auto &d = calibrationData;
    for (int i = 0; i < imagePoints.size(); ++i) {
        cv::undistortPoints(imagePoints[i], plainPoints[i], d.cameraMatrix, d.distCoeff, cv::noArray(), d.cameraMatrix);
    }
}

void SingleCameraThread::undistortImagePoints(const std::vector<ecv::CalibrateMapper::Point3> &imagePoints, std::vector<ecv::CalibrateMapper::Point3> &plainPoints, const CalibrationData &calibrationData) const {
    cv::Mat ip, pp((int)imagePoints.size(), 1, CV_32FC2);
    converPoints(imagePoints, ip);
    std::vector vpp = {pp};
    undistortImagePoints({ip}, vpp, calibrationData);
    converPoints(pp, plainPoints);
}

void SingleCameraThread::converPoints(const cv::Mat &pp, std::vector<ecv::CalibrateMapper::Point3> &points) const {
    for (int i = 0; i < pp.total(); ++i) {
        const auto &p = pp.at<cv::Point2f>(i);
        points[i].x = p.x;
        points[i].y = p.y;
        points[i].z = 0.0;
    }
}


void SingleCameraThread::converPoints(const std::vector<ecv::CalibrateMapper::Point3> &points, cv::Mat &pp) const {
    pp = cv::Mat::zeros((int)points.size(), 1, CV_32FC2);
    for (int i = 0; i < points.size(); ++i) {
        pp.at<cv::Point2f>(i) = cv::Point2f((float)points[i].x, (float)points[i].y);
    }
}