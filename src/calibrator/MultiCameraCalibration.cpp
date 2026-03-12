#include <iomanip>
#include <calib3d.hpp>
#include <random>
#include "MultiCameraCalibration.h"
#include "StatStreaming.h"
#include "CalibrateMapper.h"
#include "CalibrationStrategy.h"

using namespace ecv;

void MultiCameraCalibration::multicamThreadCallback(const std::vector<FrameRefList> &frameSets) {
    std::vector<FrameRefList> validFrameSets;
    std::vector<FrameRefList> trainFrameSets;

    size_t w, h;
    if (frameSets.empty()) {
        std::tie(w, h) = gridPreferredSizeProvider.getGridPreferredSize();
    } else {
        w = frameSets.begin()->begin()->get()->w;
        h = frameSets.begin()->begin()->get()->h;
    }

    for (const auto &frameSet : frameSets) {
        if (isValid(frameSet) && frameCollector.addMulticamFrameSet(frameSet)) {
            validFrameSets.emplace_back(frameSet);
            if (!frameSet[0]->validate) {
                trainFrameSets.emplace_back(frameSet);
            }
        }
    }

    auto sample = frameCollector.getFrameSetsSample(TRAIN_SAMPLE_SIZE, w, h, false);

    for (const auto &frameSet : sample) {
        trainFrameSets.emplace_back(frameSet);
    }

    if (trainFrameSets.empty()) {
        return;
    }

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
        flagsForIntrinsics.emplace_back(cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_FIX_INTRINSIC);
        Ks.emplace_back(calibrationData.cameraMatrix);
        Rs.emplace_back(calibrationData.Rs);
        Ts.emplace_back(calibrationData.Ts);
        distortions.emplace_back(calibrationData.distCoeff, CV_64FC1);
    }

    auto mask = cv::Mat(numCameras, (int)trainFrameSets.size(), CV_8U);
    objectPoints = getObjectPointsFromFrameSets(trainFrameSets);
    imagePoints = getImagePointsFromFrameSets(trainFrameSets);

    for (int frameId = 0; frameId < trainFrameSets.size(); frameId++) {
        for (int camId = 0; camId < trainFrameSets[frameId].size(); camId++) {
            mask.at<unsigned char>(camId, frameId) = trainFrameSets[frameId][camId] == nullptr ? 0 : 255;
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
                cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_FIX_INTRINSIC | cv::CALIB_USE_EXTRINSIC_GUESS,
                cv::TermCriteria(10, 1e-8)
        );
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return;
    }

    const auto &sampleValidate = frameCollector.getFrameSetsSample(VALIDATE_SAMPLE_SIZE, w, h, true);

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
            mask2.at<unsigned char>(camId, frameId) = sampleValidate[frameId][camId] == nullptr ? 0 : 255;
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
        Rs2[i] = Rs[i].clone();
        Ts2[i] = Ts[i].clone();
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
                cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_USE_EXTRINSIC_GUESS | cv::CALIB_FIX_INTRINSIC,
                cv::TermCriteria(1, 1e-7)
        );
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return;
    }

    if (cost > 10) {
        return;
    }

    for (int i = 1; i < numCameras; ++i) {
        viewMulticamCosts[i] = cost;
        cv::Mat tmp;
        auto calibrationData = getCalibrationData(i);
        auto calibrationData0 = getCalibrationData(0);
        calibrationData.cameraMatrix = Ks[i];
        calibrationData.distCoeff = distortions[i];
        setCalibrationData(i, calibrationData);
        cv::Mat R;

        cv::Rodrigues(Rs[i], R);

        cv::Mat R1, R2, P1, P2, Q;
        cv::Rect roi1, roi2;

        cv::stereoRectify(
                Ks[0], distortions[0],
                Ks[i], distortions[i],
                frameSize,
                R, Ts[i],
                R1, R2, P1, P2, Q,
                cv::CALIB_ZERO_DISPARITY,
                0.1, // alpha
                frameSize, &roi1, &roi2
        );

//        multicamRoiSize = roi2.width * roi2 .height;

        calibrationData.Ri = R2;
        calibrationData.Pi = P2;
        calibrationData.Rs = Rs[i];
        calibrationData.Ts = Ts[i];
        calibrationData0.Ri = R1;
        calibrationData0.Pi = P1;
        calibrationData0.Rs = Rs[0];
        calibrationData0.Ts = Ts[0];

        std::uniform_real_distribution<double> randomRange(0., 1.);
        std::mt19937 r {std::random_device{}()};
        double annealEma = 2. / (20. + 1.);

        auto gridCost = verifyParamsUsingGridMatch(imagePoints[i], calibrationData);
        auto gridCost0 = verifyParamsUsingGridMatch(imagePoints[0], calibrationData0);

        // оптимизация методом адаптивных ограничений
        bool condition = (gridCost < gridDistanceCosts[i] || gridCost0 < gridDistanceCosts[0]) && cost < multicamCosts * 2;
        if (condition || randomRange(r) < std::pow(std::numbers::e, (gridDistanceCosts[i] - gridCost) / temperature)) {
            std::cout << (condition ? "Normal... " : "Annealing... ") << i << std::endl;

            printMulticamCalibrationStats(Ks, Rs, Ts, "c");
            gridDistanceCosts[i] = std::min(gridCost, gridDistanceCosts[i]);
            multicamCosts = std::min(cost, multicamCosts);

            setRectificationData(i, calibrationData);
            setRectificationData(0, calibrationData0);

            cv::initUndistortRectifyMap(Ks[0], distortions[0], R1, P1, frameSize, CV_32FC2, rectifiedMap[0], tmp);
            cv::initUndistortRectifyMap(Ks[i], distortions[i], R2, P2, frameSize, CV_32FC2, rectifiedMap[i], tmp);

            onUpdateCallback(0, *this);
            onUpdateCallback(i, *this);

            annealFreq = annealEma * 1 + (1 - annealEma) * annealFreq;
        } else {
            annealFreq = annealEma * 0 + (1 - annealEma) * annealFreq;
        }

        if (annealFreq > 0.8 && temperature > 1e-2) {
            std::cout << "Cooling... " << i << std::endl;
            temperature /= 1.1;
        } else if (annealFreq < 0.2 && temperature < 1e2) {
            std::cout << "Heating... " << i << std::endl;
            temperature *= 1.1;
        }
    }
}

/**
 * Вычисляет корреляцию между сеткой исходного изображения и исправленного.
 * Так как камера может обладать значительной дисторсией, то в норме отклонения
 * могут быть очень большими и не показательными. Можно лишь утверждать что
 * даже fisheye объектив имеет область где такие искажения минимальны.
 * Поэтому если мы возьмем  несколько самых близких точек, то получим близкое к нулю значение
 * если кадры хорошо сопоставлены, и огромное значение, если есть большой сдвиг или поворот.
 *
 * @return average of shortest 10 distances between undistorted and image point
 */
double MultiCameraCalibration::verifyParamsUsingGridMatch(const std::vector<cv::Mat> &imagePoints, const CalibrationData &calibrationData) const {
    std::multiset<double> distances;
    std::vector<cv::Mat> plainPoints(imagePoints.size());
    rectifyImagePoints(imagePoints, plainPoints, calibrationData);
    for (int i = 0; i < imagePoints.size(); ++i) {
        for (int j = 0; j < plainPoints[i].total(); ++j) {
            auto plain = plainPoints[i].at<cv::Point2f>(j);
            auto src = imagePoints[i].at<cv::Point2f>(j);
            double d = std::sqrt(std::pow(plain.x - src.x, 2.) + std::pow(plain.y - src.y, 2.));
            distances.insert(d);
        }
    }

    StatStreaming stat;

    stat.addFirstValue(0);

    for (auto d : distances | std::views::take(distances.size() / 2)) {
        stat.addValue(d);
    }

    std::cout << "Соответствие кадров: расстояние между сетками " << stat.mean() << " отклонение " << stat.stddev() << std::endl;

    return stat.mean() * 0.3 + stat.stddev() * 0.7;
}

void MultiCameraCalibration::rectifyImagePoints(const std::vector<cv::Mat> &imagePoints, std::vector<cv::Mat> &plainPoints, const CalibrationData &calibrationData) {
    std::multiset<double> distances;
    const auto &d = calibrationData;
    for (int i = 0; i < imagePoints.size(); ++i) {
        cv::undistortPoints(imagePoints[i], plainPoints[i], d.cameraMatrix, d.distCoeff, d.Ri, d.Pi);
    }
}

void MultiCameraCalibration::rectifyImagePoints(const std::vector<ecv::CalibrateMapper::Point3> &imagePoints, std::vector<ecv::CalibrateMapper::Point3> &plainPoints, const CalibrationData &calibrationData) {
    cv::Mat ip, pp((int)imagePoints.size(), 1, CV_32FC2);
    CalibrationStrategy::converPoints(imagePoints, ip);
    std::vector vpp = {pp};
    rectifyImagePoints({ip}, vpp, calibrationData);
    CalibrationStrategy::converPoints(pp, plainPoints);
}

std::vector<cv::Mat> MultiCameraCalibration::getObjectPointsFromFrameSets(const std::vector<FrameRefList> &frameSets) const {
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

bool MultiCameraCalibration::isValid(const FrameRefList &frameSet) const {
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

std::vector<std::vector<cv::Mat>> MultiCameraCalibration::getImagePointsFromFrameSets(const std::vector<FrameRefList> &frameSets) const {
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
void MultiCameraCalibration::printMulticamCalibrationStats(
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
        cv::Vec3d rvec = R;
        double angle = cv::norm(rvec);
        double angle_deg = angle * 180.0 / CV_PI;
        cv::Vec3d axis = rvec / (angle + 1e-12);

        std::cout << "  Вращение R (от камеры 0 к камере " << i << "):\n";
        std::cout << "    Угол поворота: " << angle_deg << "°\n";
        std::cout << "    Ось поворота: [ " << axis[0] << ", " << axis[1] << ", " << axis[2] << " ]\n";
        if (angle_deg < 10.0) {
            std::cout << "    Разложение по осям (приблизительно):\n";
            std::cout << "      pitch (X): " << rvec[0] * 180.0 / CV_PI << "°\n";
            std::cout << "      yaw   (Y): " << rvec[1] * 180.0 / CV_PI << "°\n";
            std::cout << "      roll  (Z): " << rvec[2] * 180.0 / CV_PI << "°\n";
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
        double sum = 0.0, sqSum = 0.0;
        for (double x : v) {
            sum += x;
            sqSum += x * x;
        }
        double mean = sum / (double)v.size();
        double stddev = std::sqrt(sqSum / (double)v.size() - mean * mean);
        return {mean, stddev};
    };

    auto [fx_mean, fx_std] = meanStd(fx_all);
    auto [fy_mean, fy_std] = meanStd(fy_all);
    auto [cx_mean, cx_std] = meanStd(cx_all);
    auto [cy_mean, cy_std] = meanStd(cy_all);

    std::cout << "\n--- Сводная статистика внутренних параметров по всем камерам ---\n";
    std::cout << "         Среднее   Стд. откл.   Отклонение от среднего (%)\n";
    std::cout << "fx     : " << std::setw(8) << fx_mean << "   " << std::setw(8) << fx_std
              << "   " << std::setw(8) << (fx_std / fx_mean * 100) << "\n";
    std::cout << "fy     : " << std::setw(8) << fy_mean << "   " << std::setw(8) << fy_std
              << "   " << std::setw(8) << (fy_std / fy_mean * 100) << "\n";
    std::cout << "cx     : " << std::setw(8) << cx_mean << "   " << std::setw(8) << cx_std << "\n";
    std::cout << "cy     : " << std::setw(8) << cy_mean << "   " << std::setw(8) << cy_std << "\n";

    std::cout << "\n========== Конец анализа ==========\n\n";
}