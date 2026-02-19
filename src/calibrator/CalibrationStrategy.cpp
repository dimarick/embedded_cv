#include <iomanip>
#include "CalibrationStrategy.h"

using namespace ecv;

void CalibrationStrategy::runCalibration() {
    running = true;

    for (int i = 0; i < camThreads.size(); i++) {
        camThreads[i] = std::thread([](decltype(this) that, int i) {
            FrameRefList pendingFrames(MAX_FRAMES_QUEUE);
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
        std::vector<FrameRefList> frames;
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

void CalibrationStrategy::addFrameSet(const FrameRefList &frameSet) {
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

void CalibrationStrategy::camThreadCallback(const FrameRefList &frames, int cameraId) {
    int i = cameraId;

    auto preferredSize = gridPreferredSizeProvider.getGridPreferredSize();
    auto w = preferredSize != nullptr ? preferredSize->w : 0;
    auto h = preferredSize != nullptr ? preferredSize->h : 0;

    auto sample = frameCollectors[i].getFramesSample(SAMPLE_SIZE, w, h, false);

    for (const auto &frame : frames) {
        if (frame != nullptr) {
            frameCollectors[i].addFrame(gridPreferredSizeProvider, frame);
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

    auto sampleValidate = frameCollectors[i].getFramesSample(SAMPLE_SIZE, w, h, true);

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

void CalibrationStrategy::multicamThreadCallback(const std::vector<FrameRefList> &frameSets) {
    std::vector<FrameRefList> validFrameSets;

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

    std::vector<cv::Mat> objectPoints;
    std::vector<std::vector<cv::Mat>> imagePoints;

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

    for (int j = 0; j < 2; ++j) {
        objectPoints = getObjectPointsFromFrameSets(validFrameSets);
        imagePoints = getImagePointsFromFrameSets(validFrameSets);

        for (int frameId = 0; frameId < validFrameSets.size(); frameId++) {
            for (int camId = 0; camId < validFrameSets[frameId].size(); camId++) {
                mask.at<unsigned char>(camId, frameId) = validFrameSets[frameId][camId] == nullptr ? 0 : 255;
            }
        }

        try {
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
                    cv::TermCriteria(1000, 1e-7)
            );
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return;
        }

        if (frameSets.size() < 3) {
            break;
        }

        auto outliers = findOutliersPerFrameError(perFrameErrors);

        auto filtered = filterOutliers(validFrameSets, outliers);

        if (filtered == 0) {
            break;
        }
    }

    frameCollectors[0].addMulticamFrames(validFrameSets);

    auto preferredSize = gridPreferredSizeProvider.getGridPreferredSize();
    auto w = preferredSize != nullptr ? preferredSize->w : 0;
    auto h = preferredSize != nullptr ? preferredSize->h : 0;

    const auto &sampleValidate = frameCollectors[0].getFrameSetsSample(VALIDATE_SAMPLE_SIZE, w, h, true);

    std::vector<cv::Mat> objectPoints2;
    std::vector<std::vector<cv::Mat>> imagePoints2;

    objectPoints2 = getObjectPointsFromFrameSets(sampleValidate);
    imagePoints2 = getImagePointsFromFrameSets(sampleValidate);

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

    double cost;
    try {
        cost = cv::calibrateMultiview(
                objectPoints2,
                imagePoints2,
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
        return;
    }

    if (cost < multicamCosts) {
        multicamCosts = cost;

        printMulticamCalibrationStats(Ks, Rs, Ts, "c");

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
            frameDataStorage[i].open(std::format("frameData{}.yaml", i), cv::FileStorage::WRITE);

            if (!frameDataStorage[i].isOpened()) {
                continue;
            }

            frameCollectors[i].store(frameDataStorage[i]);

            frameDataStorage[i].release();
        }

        for (int i = 0; i < numCameras; ++i) {
            onUpdateCallback(i, *this);
        }
    } else {
        multicamCosts *= 1.05;
    }
}

int CalibrationStrategy::filterOutliers(std::vector<FrameRefList> &frameSets, const std::vector<bool> &outliers) const {
    int i, k = 0;
    for (i = 0; i < frameSets.size(); ++i) {
        if (!outliers[i]) {
            frameSets[k] = frameSets[i];
            k++;
        } else if (i > k) {
            frameSets[k] = frameSets[i];
        }
    }

    frameSets.resize(k);

    return i - k;
}

std::vector<cv::Mat> CalibrationStrategy::getObjectPointsFromFrameSets(const std::vector<FrameRefList> &frameSets) const {
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

bool CalibrationStrategy::isValid(const FrameRefList &frameSet) const {
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

std::vector<std::vector<cv::Mat>> CalibrationStrategy::getImagePointsFromFrameSets(const std::vector<FrameRefList> &frameSets) const {
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

/**
 * @brief Выводит статистику по многокамерной калибровке: для каждой камеры относительно базовой (первой) и общий разброс параметров.
 *
 * @param camMatrices Вектор матриц камер (3x3) для всех N камер.
 * @param Rs          Вектор матриц поворота (3x3) для всех камер относительно первой.
 *                    Для первой камеры должна быть единичная матрица.
 * @param Ts          Вектор векторов трансляции (3x1) для всех камер относительно первой.
 *                    Для первой камеры должен быть нулевой вектор.
 * @param unit        Строка с обозначением единиц измерения (например, "мм").
 */
void CalibrationStrategy::printMulticamCalibrationStats(
        const std::vector<cv::Mat>& camMatrices,
        const std::vector<cv::Mat>& Rs,
        const std::vector<cv::Mat>& Ts,
        const std::string& unit = "ед.")
{
    CV_Assert(camMatrices.size() == Rs.size() && camMatrices.size() == Ts.size());
    int N = (int)camMatrices.size();
    if (N < 1) return;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n========== Многокамерная калибровка: анализ соосности ==========\n";
    std::cout << "Всего камер: " << N << "\n";
    std::cout << "Единицы измерения: " << unit << "\n";

    // Базовая камера — индекс 0
    const cv::Mat& baseCam = camMatrices[0];
    double fx0 = baseCam.at<double>(0,0);
    double fy0 = baseCam.at<double>(1,1);
    double cx0 = baseCam.at<double>(0,2);
    double cy0 = baseCam.at<double>(1,2);

    // Векторы для сбора статистики по внутренним параметрам
    std::vector<double> fx_all, fy_all, cx_all, cy_all;
    fx_all.push_back(fx0);
    fy_all.push_back(fy0);
    cx_all.push_back(cx0);
    cy_all.push_back(cy0);

    for (int i = 1; i < N; ++i) {
        std::cout << "\n--- Камера " << i << " относительно камеры 0 ---\n";

        const cv::Mat& cam = camMatrices[i];
        const cv::Mat& R = Rs[i];
        const cv::Mat& T = Ts[i];

        // Проверки размеров (можно убрать для скорости, но для отладки оставим)
        CV_Assert(cam.size() == cv::Size(3,3));
        CV_Assert(R.size() == cv::Size(1,3));
        CV_Assert(!T.empty() && T.total() == 3);

        cv::Mat RR;
        cv::Rodrigues(R, RR);
        // --- 1. Трансляция ---
        cv::Vec3d t;
        if (T.rows == 3 && T.cols == 1)
            t = cv::Vec3d(T.at<double>(0,0), T.at<double>(1,0), T.at<double>(2,0));
        else if (T.rows == 1 && T.cols == 3)
            t = cv::Vec3d(T.at<double>(0,0), T.at<double>(0,1), T.at<double>(0,2));
        else {
            CV_Error(cv::Error::StsBadArg, "T должен быть 3x1 или 1x3");
        }

        double baseline = cv::norm(t);
        std::cout << "  Трансляция T (от камеры 0 к камере " << i << "):\n";
        std::cout << "    T_x = " << t[0] << " " << unit << "\n";
        std::cout << "    T_y = " << t[1] << " " << unit << "\n";
        std::cout << "    T_z = " << t[2] << " " << unit << "\n";
        std::cout << "    Базовая линия |T| = " << baseline << " " << unit << "\n";
        if (baseline > 1e-6) {
            std::cout << "    Отклонение от горизонтальной соосности:\n";
            std::cout << "      по вертикали (T_y) : " << t[1] << " (" << (t[1] / baseline * 100) << "% от базлайна)\n";
            std::cout << "      по глубине   (T_z) : " << t[2] << " (" << (t[2] / baseline * 100) << "% от базлайна)\n";
        } else {
            std::cout << "    Базовая линия слишком мала для процентных отклонений.\n";
        }

        // --- 2. Вращение ---
        cv::Vec3d rvec;
        cv::Rodrigues(RR, rvec);
        double angle = cv::norm(rvec);
        double angle_deg = angle * 180.0 / CV_PI;
        cv::Vec3d axis = rvec / (angle + 1e-12);

        std::cout << "  Вращение R (от камеры 0 к камере " << i << "):\n";
        std::cout << "    Угол поворота: " << angle_deg << "°\n";
        std::cout << "    Ось поворота: [ " << axis[0] << ", " << axis[1] << ", " << axis[2] << " ]\n";
        if (angle_deg < 10.0) {
            std::cout << "    Разложение по осям (приблизительно):\n";
            std::cout << "      roll  (X): " << rvec[0] * 180.0 / CV_PI << "°\n";
            std::cout << "      pitch (Y): " << rvec[1] * 180.0 / CV_PI << "°\n";
            std::cout << "      yaw   (Z): " << rvec[2] * 180.0 / CV_PI << "°\n";
        } else {
            std::cout << "    Вектор Родригеса (rx, ry, rz): [ "
                      << rvec[0] << ", " << rvec[1] << ", " << rvec[2] << " ] рад\n";
        }

        // --- 3. Внутренние параметры ---
        double fx = cam.at<double>(0,0);
        double fy = cam.at<double>(1,1);
        double cx = cam.at<double>(0,2);
        double cy = cam.at<double>(1,2);

        fx_all.push_back(fx);
        fy_all.push_back(fy);
        cx_all.push_back(cx);
        cy_all.push_back(cy);

        std::cout << "  Внутренние параметры камеры " << i << " (сравнение с камерой 0):\n";
        std::cout << "            Камера 0   Камера " << i << "   Разница   Отн. разница (%)\n";
        std::cout << "fx        : " << std::setw(8) << fx0 << "   " << std::setw(8) << fx
                  << "   " << std::setw(8) << (fx - fx0) << "   "
                  << std::setw(8) << ((fx - fx0) / fx0 * 100) << "\n";
        std::cout << "fy        : " << std::setw(8) << fy0 << "   " << std::setw(8) << fy
                  << "   " << std::setw(8) << (fy - fy0) << "   "
                  << std::setw(8) << ((fy - fy0) / fy0 * 100) << "\n";
        std::cout << "cx        : " << std::setw(8) << cx0 << "   " << std::setw(8) << cx
                  << "   " << std::setw(8) << (cx - cx0) << "\n";
        std::cout << "cy        : " << std::setw(8) << cy0 << "   " << std::setw(8) << cy
                  << "   " << std::setw(8) << (cy - cy0) << "\n";
    }

    // --- 4. Общая статистика по внутренним параметрам ---
    auto meanStd = [](const std::vector<double>& v) -> std::pair<double, double> {
        double sum = 0.0, sqsum = 0.0;
        for (double x : v) {
            sum += x;
            sqsum += x * x;
        }
        double mean = sum / v.size();
        double stddev = std::sqrt(sqsum / v.size() - mean * mean);
        return {mean, stddev};
    };

    auto [fx_mean, fx_std] = meanStd(fx_all);
    auto [fy_mean, fy_std] = meanStd(fy_all);
    auto [cx_mean, cx_std] = meanStd(cx_all);
    auto [cy_mean, cy_std] = meanStd(cy_all);

    std::cout << "\n--- Сводная статистика внутренних параметров по всем камерам ---\n";
    std::cout << "         Среднее   Стд.откл.   Отклонение от среднего (%)\n";
    std::cout << "fx     : " << std::setw(8) << fx_mean << "   " << std::setw(8) << fx_std
              << "   " << std::setw(8) << (fx_std / fx_mean * 100) << "\n";
    std::cout << "fy     : " << std::setw(8) << fy_mean << "   " << std::setw(8) << fy_std
              << "   " << std::setw(8) << (fy_std / fy_mean * 100) << "\n";
    std::cout << "cx     : " << std::setw(8) << cx_mean << "   " << std::setw(8) << cx_std << "\n";
    std::cout << "cy     : " << std::setw(8) << cy_mean << "   " << std::setw(8) << cy_std << "\n";

    std::cout << "\n========== Конец анализа ==========\n\n";
}

/**
 * @brief Возвращает индексы frame элементов матрицы errors, превышающих порог mean + k * stddev.
 * @param errors  cv::Mat размером (numFrames x numCameras), тип CV_64F.
 * @param k       коэффициент для порога (по умолчанию 2.0).
 * @return std::vector<std::pair<int,int>> список пар (frame, cam)-индексов выбросов.
 */
std::vector<bool> CalibrationStrategy::findOutliersPerFrameError(const cv::Mat& errors, double k) {
    std::vector<bool> outliers(errors.cols, false);

    if (errors.empty() || errors.type() != CV_64F) {
        std::cerr << "findOutliersPerFrameError: errors empty or not double" << std::endl;
        return outliers;
    }

    // 1. Вычислим среднее и стандартное отклонение по всем элементам
    double sum = 0.0, sumSq = 0.0;
    int total = errors.rows * errors.cols;
    for (int r = 0; r < errors.rows; ++r) {
        for (int c = 0; c < errors.cols; ++c) {
            double val = errors.at<double>(r, c);
            sum += val;
            sumSq += val * val;
        }
    }


    double mean = sum / total;
    double variance = (sumSq / total) - (mean * mean);
    double stddev = std::sqrt(variance);

    double threshold = mean + k * stddev;
    std::cout << "Outlier filter: mean = " << mean << ", stddev = " << stddev
              << ", threshold = " << threshold << std::endl;

    // 2. Собираем индексы превышающих порог
    for (int cam = 0; cam < errors.rows; ++cam) {
        for (int f = 0; f < errors.cols; ++f) {
            if (errors.at<double>(cam, f) > threshold) {
                outliers[f] = true;
            }
        }
    }

    return outliers;
}

void CalibrationStrategy::loadConfig() {
//    for (int i = 0; i < numCameras; ++i) {
//        frameDataStorage[i].open(std::format("frameData{}.yaml", i), cv::FileStorage::READ);
//
//        if (!frameDataStorage[i].isOpened()) {
//            continue;
//        }
//
//        frameCollectors[i].load(gridPreferredSizeProvider, frameDataStorage[i]);
//
//        frameDataStorage[i].release();
//    }

    auto preferredSize = gridPreferredSizeProvider.getGridPreferredSize();
    auto w = preferredSize != nullptr ? preferredSize->w : 0;
    auto h = preferredSize != nullptr ? preferredSize->h : 0;
    const auto &sample = frameCollectors[0].getFrameSetsSample(1000, w, h, false);
    for (const auto &item : sample) {
        addFrameSet(item);
    }
}
