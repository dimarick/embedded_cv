#include "Calibrator.h"
#include <opencv2/calib3d.hpp>
#include <iomanip>

namespace ecv {
    double Calibrator::calibrateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3f>> &objectPoints,
                                             const std::vector<std::vector<cv::Point2f>> &imagePoints,
                                             CalibrationData &data,
                                             int flags,
                                             cv::TermCriteria term) {
        size_t objectSize = 0;
        for (const auto & objectPoint : objectPoints) {
            if (objectPoint.empty()) {
                break;
            }
            objectSize++;
        }

        size_t imageSize = 0;
        for (const auto & imagePoint : imagePoints) {
            if (imagePoint.empty()) {
                break;
            }
            imageSize++;
        }

        auto inputSize = std::min(objectSize, imageSize);

        auto objectPointsSlice = objectPoints;
        auto imagePointsSlice = imagePoints;
        if (inputSize < imagePoints.size()) {
            objectPointsSlice = std::vector(objectPoints.begin(), objectPoints.begin() + (int)inputSize);
            imagePointsSlice = std::vector(imagePoints.begin(), imagePoints.begin() + (int)inputSize);
        }

        auto cameraData = (double *)data.cameraMatrix.data;

        fx = cameraData[0] == 0 ? 1000 : cameraData[0];
        fy = cameraData[4] == 0 ? 1000 : cameraData[4];

        cx = cameraData[2] == 0 ? (float)frameSize.width / 2 : cameraData[2];
        cy = cameraData[5] == 0 ? (float)frameSize.height / 2 : cameraData[5];

        cameraData[0] = fx;
        cameraData[4] = fy;
        cameraData[2] = cx;
        cameraData[5] = cy;
        cameraData[8] = 1.;

        auto random = std::rand() % 8;

        int baseFlags =
                cv::CALIB_USE_LU|
                (cv::CALIB_RATIONAL_MODEL && random & 1)|
                (cv::CALIB_TILTED_MODEL && random & 2)|
                (cv::CALIB_THIN_PRISM_MODEL && random & 4)|
                cv::CALIB_FIX_ASPECT_RATIO|
                cv::CALIB_USE_INTRINSIC_GUESS;

        double result = 0;

        std::vector<double> distCoeff;
        for (int i = 0; i < distCoeff.size(); ++i) {
            distCoeff[i] = data.distCoeff[i];
        }

        try {
            result = cv::calibrateCamera(objectPointsSlice, imagePointsSlice, frameSize, data.cameraMatrix,
                                data.distCoeff, data.rvecs, data.tvecs, baseFlags | flags, term);
        } catch (const std::exception &e) {
            std::cerr << "cv::calibrateCamera failed" << std::endl;

            return 1. / 0.;
        }

//        std::cout << std::format("result = {}, camera.fx = {}, camera.fy = {}, camera.cx = {}, camera.cy = {}, distCoeff = {}\n",
//                                 result, cameraData[0], cameraData[4], cameraData[2], cameraData[5], data.distCoeff);

        double ema = 2. / (21. + 1.);
        fx = ema * cameraData[0] + (1 - ema) * fx;
        fy = ema * cameraData[4] + (1 - ema) * fy;
        cx = ema * cameraData[2] + (1 - ema) * cx;
        cy = ema * cameraData[5] + (1 - ema) * cy;

        for (int i = 0; i < distCoeff.size(); ++i) {
            data.distCoeff[i] = ema * data.distCoeff[i] + (1 - ema) * distCoeff[i];
        }

        return result;
    }

    std::tuple<cv::Mat, cv::Mat, double> Calibrator::getStereoUndistortMap(cv::Size frameSize, const CalibrationData &base, const CalibrationData &current) {
        cv::Rect roi1, roi2;
        cv::Mat  baseMap, map, tmp, R1, P1, R2, P2, Q;

        double alpha = 1.0;
//        cv::Mat newBaseCamMat = cv::getOptimalNewCameraMatrix(base.cameraMatrix, base.distCoeff, frameSize, alpha, frameSize, &roi1);
//        cv::Mat newCamMat = cv::getOptimalNewCameraMatrix(current.cameraMatrix, current.distCoeff, frameSize, alpha, frameSize, &roi2);
//        cv::Mat avgMat = (newBaseCamMat + newCamMat) / 2;

        cv::stereoRectify(
                base.cameraMatrix, base.distCoeff,
                current.cameraMatrix, current.distCoeff,
                frameSize, current.R, current.T,
                R1, R2, P1, P2, Q,
                cv::CALIB_ZERO_DISPARITY, alpha, frameSize, &roi1, &roi2
        );
        cv::Mat rvec;
        cv::Rodrigues(current.R, rvec);

//        std::cout << "rvec (rad): " << rvec.t() << std::endl;
//        std::cout << "T (mm): " << current.T.t() << std::endl;
//
//        std::cout << "cam1:\n" << base.cameraMatrix << std::endl;
//        std::cout << "cam2:\n" << current.cameraMatrix << std::endl;
//
//        std::cout << "P1:\n" << P1 << std::endl;
//        std::cout << "P2:\n" << P2 << std::endl;
//        std::cout << "R1:\n" << R1 << std::endl;
//        std::cout << "R2:\n" << R2 << std::endl;

//        printStereoCalibrationStats(base.cameraMatrix, current.cameraMatrix, current.R, current.T);

        cv::initUndistortRectifyMap(base.cameraMatrix, base.distCoeff, R2, P2, frameSize, CV_32FC2, baseMap, tmp);
//        cv::stereoRectify(
//                base.cameraMatrix, base.distCoeff,
//                current.cameraMatrix, current.distCoeff,
//                frameSize, R1.t() * R2, current.T,
//                R1, R2, P1, P2, Q,
//                cv::CALIB_ZERO_DISPARITY, 0.0, frameSize, &roi1, &roi2
//        );

        cv::initUndistortRectifyMap(current.cameraMatrix, current.distCoeff, R1, P1, frameSize, CV_32FC2, map, tmp);

        return {baseMap, map, roi1.width * roi1.height};
    }

    void Calibrator::convertTo2dPoints(const std::vector<cv::Point3d> &points3d, std::vector<cv::Point2f> &points2d) {
        for (int j = 0; j < points2d.size(); ++j) {
            points2d[j] = cv::Point(points3d[j].x, points3d[j].y);
        }
    }

    void Calibrator::convertToPlain3dPoints(const std::vector<cv::Point3d> &points1, std::vector<cv::Point3f> &points2) {
        for (int j = 0; j < points2.size(); ++j) {
            points2[j] = cv::Point3f(points1[j].x, points1[j].y, 0);
        }
    }

    void Calibrator::convertTo2dPoints(const std::vector<cv::Point3f> &points3d, std::vector<cv::Point2f> &points2d) {
        for (int j = 0; j < points2d.size(); ++j) {
            points2d[j] = cv::Point(points3d[j].x, points3d[j].y);
        }
    }

    double Calibrator::calibrateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3d>> &collectedObjectPoints,
                                             const std::vector<std::vector<cv::Point3d>> &collectedImagePoints,
                                             const std::vector<cv::Point3d> &newObjectPoints,
                                             const std::vector<cv::Point3d> &newImagePoints,
                                             CalibrationData &data,
                                             int flags,
                                             cv::TermCriteria term) {

        cv::Mat mat;
        std::vector<std::vector<cv::Point3d>> objectGrids(collectedObjectPoints.size());
        std::vector<std::vector<cv::Point3d>> imageGrids(collectedImagePoints.size());
        std::copy(collectedImagePoints.begin(), collectedImagePoints.end(), imageGrids.begin());
        std::copy(collectedObjectPoints.begin(), collectedObjectPoints.end(), objectGrids.begin());
        objectGrids.emplace_back(newObjectPoints);
        imageGrids.emplace_back(newImagePoints);

        return calibrateSingleCamera(frameSize, objectGrids, imageGrids, data, flags, term);
    }

    double Calibrator::calibrateSingleCamera(cv::Size frameSize, const std::vector<std::vector<cv::Point3d>> &objectPoints,
                                             const std::vector<std::vector<cv::Point3d>> &imagePoints,
                                             CalibrationData &data,
                                             int flags,
                                             cv::TermCriteria term) {
        std::vector<std::vector<cv::Point3f>> objectPoints2(objectPoints.size());
        std::vector<std::vector<cv::Point2f>> imagePoints2(imagePoints.size());

        for (int i = 0; i < imagePoints.size(); ++i) {
            imagePoints2[i].resize(imagePoints[i].size());
            convertTo2dPoints(imagePoints[i], imagePoints2[i]);
        }

        for (int i = 0; i < objectPoints.size(); ++i) {
            objectPoints2[i].resize(objectPoints[i].size());
            convertToPlain3dPoints(objectPoints[i], objectPoints2[i]);
        }

        return calibrateSingleCamera(frameSize, objectPoints2, imagePoints2, data, flags, term);
    }

    double Calibrator::calibrateCameraPair(
            cv::Size frameSize,
            const std::vector<std::vector<cv::Point3d>> &objectPointsCam1,
            const std::vector<std::vector<cv::Point3d>> &imagePointsCam1,
            const std::vector<std::vector<cv::Point3d>> &objectPoints,
            const std::vector<std::vector<cv::Point3d>> &imagePoints,
            CalibrationData &dataCam1,
            CalibrationData &data,
            cv::TermCriteria term) {

        std::vector<std::vector<cv::Point3f>> _objectPointsCam1(objectPointsCam1.size());
        std::vector<std::vector<cv::Point2f>> _imagePointsCam1(imagePointsCam1.size());
        std::vector<std::vector<cv::Point3f>> _objectPoints(objectPoints.size());
        std::vector<std::vector<cv::Point2f>> _imagePoints(imagePoints.size());

        for (int i = 0; i < imagePointsCam1.size(); ++i) {
            _imagePointsCam1[i].resize(imagePointsCam1[i].size());
            convertTo2dPoints(imagePointsCam1[i], _imagePointsCam1[i]);
        }

        for (int i = 0; i < imagePoints.size(); ++i) {
            _imagePoints[i].resize(imagePoints[i].size());
            convertTo2dPoints(imagePoints[i], _imagePoints[i]);
        }

        for (int i = 0; i < objectPointsCam1.size(); ++i) {
            _objectPointsCam1[i].resize(objectPointsCam1[i].size());
            convertToPlain3dPoints(objectPointsCam1[i], _objectPointsCam1[i]);
        }

        for (int i = 0; i < objectPoints.size(); ++i) {
            _objectPoints[i].resize(objectPoints[i].size());
            convertToPlain3dPoints(objectPoints[i], _objectPoints[i]);
        }

        return calibrateCameraPair(frameSize, _objectPointsCam1, _imagePointsCam1, _objectPoints, _imagePoints,
                                   dataCam1, data, term);
    }

    double Calibrator::calibrateCameraPair(
            cv::Size frameSize,
            const std::vector<CalibrateFrameCollector::FramePairRef> &pairs,
            const std::vector<cv::Point3d> &objectPointsCam1,
            const std::vector<cv::Point3d> &imagePointsCam1,
            const std::vector<cv::Point3d> &objectPoints,
            const std::vector<cv::Point3d> &imagePoints,
            CalibrationData &dataCam1,
            CalibrationData &data,
            cv::TermCriteria term) {

        std::vector<std::vector<cv::Point3d>> imageGrids0, imageGrids1;
        std::vector<std::vector<cv::Point3d>> objectGrids0, objectGrids1;

        for (const auto &pair : pairs) {
            const auto &baseFrame = pair->base;
            const auto &frame = pair->current;

            imageGrids0.emplace_back(baseFrame->imageGrid);
            imageGrids1.emplace_back(frame->imageGrid);

            objectGrids0.emplace_back(baseFrame->objectGrid);
            objectGrids1.emplace_back(frame->objectGrid);
        }

        imageGrids0.emplace_back(imagePointsCam1);
        imageGrids1.emplace_back(imagePoints);

        objectGrids0.emplace_back(objectPointsCam1);
        objectGrids1.emplace_back(objectPoints);

        return calibrateCameraPair(frameSize, objectGrids0, imageGrids0, objectGrids1, imageGrids1,
                                   dataCam1, data, term);
    }

    double Calibrator::calibrateCameraPair(
            cv::Size frameSize,
            const std::vector<CalibrateFrameCollector::FramePairRef> &pairs,
            CalibrationData &dataCam1,
            CalibrationData &data,
            cv::TermCriteria term) {

        std::vector<std::vector<cv::Point3d>> imageGrids0, imageGrids1;
        std::vector<std::vector<cv::Point3d>> objectGrids0, objectGrids1;

        for (const auto &pair : pairs) {
            const auto &baseFrame = pair->base;
            const auto &frame = pair->current;

            imageGrids0.emplace_back(baseFrame->imageGrid);
            imageGrids1.emplace_back(frame->imageGrid);

            objectGrids0.emplace_back(baseFrame->objectGrid);
            objectGrids1.emplace_back(frame->objectGrid);
        }

        return calibrateCameraPair(frameSize, objectGrids0, imageGrids0, objectGrids1, imageGrids1,
                                   dataCam1, data, term);
    }

    double Calibrator::calibrateCameraPair(
            cv::Size frameSize,
            const std::vector<std::vector<cv::Point3f>> &objectPointsCam1,
            const std::vector<std::vector<cv::Point2f>> &imagePointsCam1,
            const std::vector<std::vector<cv::Point3f>> &objectPoints,
            const std::vector<std::vector<cv::Point2f>> &imagePoints,
            CalibrationData &dataCam1,
            CalibrationData &data,
            cv::TermCriteria term) {

        cv::Mat perViewErrors;

        auto cost = cv::registerCameras(objectPoints, objectPointsCam1, imagePoints, imagePointsCam1,
            data.cameraMatrix, data.distCoeff, cv::CameraModel::CALIB_MODEL_PINHOLE,
            dataCam1.cameraMatrix, dataCam1.distCoeff, cv::CameraModel::CALIB_MODEL_PINHOLE,
            data.R, data.T, data.E, data.F, perViewErrors, cv::CALIB_FIX_INTRINSIC, term);

        return cost;
    }

    /**
     * @brief Выводит в std::cout параметры взаимного расположения камер и сравнивает их внутренние параметры.
     *
     * @param camMatrixL  Матрица левой камеры (3x3)
     * @param camMatrixR  Матрица правой камеры (3x3)
     * @param R           Матрица вращения правой камеры относительно левой (3x3)
     * @param T           Вектор трансляции правой камеры относительно левой (3x1 или 1x3)
     */
    void Calibrator::printStereoCalibrationStats(const cv::Mat& camMatrixL, const cv::Mat& camMatrixR,
                                     const cv::Mat& R, const cv::Mat& T) {
        CV_Assert(!camMatrixL.empty() && camMatrixL.size() == cv::Size(3, 3));
        CV_Assert(!camMatrixR.empty() && camMatrixR.size() == cv::Size(3, 3));
        CV_Assert(!R.empty() && R.size() == cv::Size(3, 3));
        CV_Assert(!T.empty() && (T.total() == 3));

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n========== Стерео-калибровка: анализ соосности ==========\n";

        // --- 1. Трансляция (смещение правой камеры относительно левой) ---
        cv::Vec3d t;
        if (T.rows == 3 && T.cols == 1)
            t = cv::Vec3d(T.at<double>(0,0), T.at<double>(1,0), T.at<double>(2,0));
        else if (T.rows == 1 && T.cols == 3)
            t = cv::Vec3d(T.at<double>(0,0), T.at<double>(0,1), T.at<double>(0,2));
        else
            CV_Error(cv::Error::StsBadArg, "T должен быть 3x1 или 1x3");

        double baseline = cv::norm(t);
        std::cout << "\n--- Трансляция (T, мм / ваши единицы) ---\n";
        std::cout << "  T_x = " << t[0] << "\n  T_y = " << t[1] << "\n  T_z = " << t[2] << "\n";
        std::cout << "  Базовая линия (|T|) = " << baseline << "\n";
        std::cout << "  Отклонение от горизонтальной соосности:\n";
        std::cout << "    по вертикали (T_y) : " << t[1] << " (" << (t[1] / baseline * 100) << "% от базлайна)\n";
        std::cout << "    по глубине   (T_z) : " << t[2] << " (" << (t[2] / baseline * 100) << "% от базлайна)\n";

        // --- 2. Вращение (отклонение ориентации правой камеры) ---
        cv::Vec3d rvec;
        cv::Rodrigues(R, rvec);
        double angle = cv::norm(rvec);               // угол поворота в радианах
        double angle_deg = angle * 180.0 / CV_PI;    // в градусах
        cv::Vec3d axis = rvec / (angle + 1e-12);     // ось поворота (единичный вектор)

        std::cout << "\n--- Вращение (R) ---\n";
        std::cout << "  Угол поворота: " << angle_deg << "°\n";
        std::cout << "  Ось поворота: [ " << axis[0] << ", " << axis[1] << ", " << axis[2] << " ]\n";
        std::cout << "  Разложение по осям (приблизительно):\n";
        // приближённое разложение на компоненты (для малых углов)
        if (angle_deg < 10.0) {
            std::cout << "    roll  (вращение вокруг X) : " << rvec[0] * 180.0 / CV_PI << "°\n";
            std::cout << "    pitch (вращение вокруг Y) : " << rvec[1] * 180.0 / CV_PI << "°\n";
            std::cout << "    yaw   (вращение вокруг Z) : " << rvec[2] * 180.0 / CV_PI << "°\n";
        } else {
            // для больших углов просто выводим вектор Родригеса
            std::cout << "  Вектор Родригеса (rx, ry, rz): [ "
                      << rvec[0] << ", " << rvec[1] << ", " << rvec[2] << " ] рад\n";
        }

        // --- 3. Сравнение внутренних параметров (масштаб и смещение главной точки) ---
        double fxL = camMatrixL.at<double>(0,0);
        double fyL = camMatrixL.at<double>(1,1);
        double cxL = camMatrixL.at<double>(0,2);
        double cyL = camMatrixL.at<double>(1,2);

        double fxR = camMatrixR.at<double>(0,0);
        double fyR = camMatrixR.at<double>(1,1);
        double cxR = camMatrixR.at<double>(0,2);
        double cyR = camMatrixR.at<double>(1,2);

        std::cout << "\n--- Внутренние параметры (cameraMatrix) ---\n";
        std::cout << "            Левая камера   Правая камера   Разница (Правая - Левая)  Отн. разница (%)\n";
        std::cout << "fx        : " << std::setw(10) << fxL << "   " << std::setw(10) << fxR
                  << "   " << std::setw(10) << (fxR - fxL) << "   "
                  << std::setw(10) << ((fxR - fxL) / fxL * 100) << "\n";
        std::cout << "fy        : " << std::setw(10) << fyL << "   " << std::setw(10) << fyR
                  << "   " << std::setw(10) << (fyR - fyL) << "   "
                  << std::setw(10) << ((fyR - fyL) / fyL * 100) << "\n";
        std::cout << "cx        : " << std::setw(10) << cxL << "   " << std::setw(10) << cxR
                  << "   " << std::setw(10) << (cxR - cxL) << "\n";
        std::cout << "cy        : " << std::setw(10) << cyL << "   " << std::setw(10) << cyR
                  << "   " << std::setw(10) << (cyR - cyL) << "\n";

        // --- 4. Соотношение фокусных расстояний (дополнительная метрика масштаба) ---
        double fx_ratio = fxR / fxL;
        double fy_ratio = fyR / fyL;
        std::cout << "\n--- Отношение фокусных расстояний (Правая / Левая) ---\n";
        std::cout << "  fx_ratio = " << fx_ratio << "\n";
        std::cout << "  fy_ratio = " << fy_ratio << "\n";
        if (std::abs(fx_ratio - 1.0) > 0.05 || std::abs(fy_ratio - 1.0) > 0.05) {
            std::cout << "  ⚠️  Заметная разница в масштабе (>5%) — рекомендуется унификация.\n";
        }

        std::cout << "\n========== Конец анализа ==========\n\n";
    }
} // ecv