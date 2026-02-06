#include <3d.hpp>
#include "CalibrateFrameCollector.h"

using namespace ecv;

template <typename T> void appendIf(std::vector<T> &vector, T value, bool condition) {
    if (condition) {
        vector.push_back(value);
    }
}

double distance(cv::Point2d p1, cv::Point2d p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

// Быстрая проверка с использованием Bounding Box
bool isInsideQuadSimple(const cv::Point2d& p, const std::vector<cv::Point2d>& quad) {
    // Считаем сумму углов - если 360°, то точка внутри
    double angleSum = 0;

    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;

        // Векторы от точки к вершинам
        double dx1 = quad[i].x - p.x;
        double dy1 = quad[i].y - p.y;
        double dx2 = quad[j].x - p.x;
        double dy2 = quad[j].y - p.y;

        // Угол между векторами через скалярное произведение
        double dot = dx1 * dx2 + dy1 * dy2;
        double cross = dx1 * dy2 - dy1 * dx2;
        angleSum += fabs(atan2(cross, dot));
    }

    // Если сумма углов ~±2π, точка внутри
    return fabs(angleSum - 2 * 3.14) < 0.1;
}

double getIntersectionAreaSize(const std::vector<cv::Point2d> &quad, const cv::Rect2d &rect) {
    // Используем равномерную сетку 20x20 = 400 точек
    const int grid = 20;  // Всего 400 точек, точность ~5%
    const double cellW = (double)rect.width / grid;
    const double cellH = (double)rect.height / grid;
    const auto total = grid * grid;
    auto hits = 0;

    for (int i = 0; i < grid; i++) {
        for (int j = 0; j < grid; j++) {
            cv::Point2d p;
            p.x = rect.x + (i + 0.5) * cellW;  // Центр ячейки
            p.y = rect.y + (j + 0.5) * cellH;

            if (isInsideQuadSimple(p, quad)) {
                hits++;
            }
        }
    }

    return (double)hits / total;
};

std::vector<CalibrateFrameCollector::FrameClass> CalibrateFrameCollector::getClasses(const CalibrateFrameCollector::Frame &frame) const {
    int w = frameSize.width;
    int h = frameSize.height;
    int w2 = w / 2;
    int h2 = h / 2;
    int w4 = w / 4;
    int h4 = h / 4;
    auto rTopLeft = cv::Rect(0, 0, w2, h2);
    auto rTopRight = cv::Rect(w2, 0, w2, h2);
    auto rBottomLeft = cv::Rect(0, h2, w2, h2);
    auto rBottomRight = cv::Rect(w2, h2, w2, h2);
    auto rCenter = cv::Rect(w4, h4, w2, h2);
    auto rAll = cv::Rect(0, 0, w, h);

    const auto &g = frame.imageGrid;
    const auto gw = frame.w;
    const auto gh = frame.h;
    const auto gw2 = gw / 2;
    const auto gh2 = gh / 2;

    cv::Point2d topLeft = {g[0].x, g[0].y};
    cv::Point2d topRight = {g[gw - 1].x, g[gw - 1].y};
    cv::Point2d bottomLeft = {g[(gh - 1) * gw + 0].x, g[(gh - 1) * gw + 0].y};
    cv::Point2d bottomRight = {g[(gh - 1) * gw + gw - 1].x, g[(gh - 1) * gw + gw - 1].y};

    std::vector<cv::Point2d> gridQuad = {
        topLeft,
        topRight,
        bottomRight,
        bottomLeft,
    };

    std::vector<CalibrateFrameCollector::FrameClass> result = {};

    auto allMatch = getIntersectionAreaSize(gridQuad, rAll);

    double yaw = 0, roll = 0, pitch = 0;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) <<
            800, 0, w2,
            0, 800, h2,
            0, 0, 1);

    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);  // Без дисторсии
    std::vector<double> rvec(3), tvec(3);

    std::vector<cv::Point3d> objectPoints = {
            {g[0].x, g[0].y, 0},
            {g[gw - 1].x, g[gw - 1].y, 0},
            {g[(gh - 1) * gw + 0].x, g[(gh - 1) * gw + 0].y, 0},
            {g[(gh - 1) * gw + gw - 1].x, g[(gh - 1) * gw + gw - 1].y, 0},
    };
    std::vector<cv::Point2d> imagePoints = {
            {0, 0},
            {(double)gw - 1, 0},
            {0, (double)gh - 1},
            {(double)gw - 1, (double)gh - 1},
    };

    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);  // Формула Родригеса

    // 5. Извлекаем углы Эйлера из матрицы вращения
    // Порядок вращений: Z (yaw), Y (pitch), X (roll)
    double sy = sqrt(rotationMatrix.at<double>(0,0) * rotationMatrix.at<double>(0,0) +
                     rotationMatrix.at<double>(1,0) * rotationMatrix.at<double>(1,0));

    bool singular = sy < 1e-6;

    if (!singular) {
        roll = atan2(rotationMatrix.at<double>(2,1),
                            rotationMatrix.at<double>(2,2));
        pitch = atan2(-rotationMatrix.at<double>(2,0), sy);
        yaw = atan2(rotationMatrix.at<double>(1,0),
                           rotationMatrix.at<double>(0,0));
    } else {
        roll = atan2(-rotationMatrix.at<double>(1,2),
                            rotationMatrix.at<double>(1,1));
        pitch = atan2(-rotationMatrix.at<double>(2,0), sy);
        yaw = 0;
    }

    // 6. Конвертируем радианы в градусы
    roll = roll * 180.0 / CV_PI;
    pitch = pitch * 180.0 / CV_PI;
    yaw = yaw * 180.0 / CV_PI;

    std::cerr << std::format("{},{},{}", roll, pitch, yaw) << std::endl;

    appendIf(result, FrameClass::topLeft, getIntersectionAreaSize(gridQuad, rTopLeft) > RECT_MATCH_THRESHOLD);
    appendIf(result, FrameClass::topRight, getIntersectionAreaSize(gridQuad, rTopRight) > RECT_MATCH_THRESHOLD);
    appendIf(result, FrameClass::bottomLeft, getIntersectionAreaSize(gridQuad, rBottomLeft) > RECT_MATCH_THRESHOLD);
    appendIf(result, FrameClass::bottomRight, getIntersectionAreaSize(gridQuad, rBottomRight) > RECT_MATCH_THRESHOLD);
    appendIf(result, FrameClass::center, getIntersectionAreaSize(gridQuad, rCenter) > RECT_MATCH_THRESHOLD);

    appendIf(result, FrameClass::near, allMatch > NEAR_THRESHOLD);
    appendIf(result, FrameClass::mid, allMatch <= NEAR_THRESHOLD && allMatch > FAR_THRESHOLD);
    appendIf(result, FrameClass::far, allMatch <= FAR_THRESHOLD);

    appendIf(result, FrameClass::rollSmall, roll > 0 && roll < SMALL_ANGLE_TAN);
    appendIf(result, FrameClass::rollMedium, roll >= SMALL_ANGLE_TAN && roll < LARGE_ANGLE_TAN);
    appendIf(result, FrameClass::rollLarge, roll >= LARGE_ANGLE_TAN);

    appendIf(result, FrameClass::pitchSmall, pitch > 0 && pitch < SMALL_ANGLE_TAN);
    appendIf(result, FrameClass::pitchMedium, pitch >= SMALL_ANGLE_TAN && pitch < LARGE_ANGLE_TAN);
    appendIf(result, FrameClass::pitchLarge, pitch >= LARGE_ANGLE_TAN);

    appendIf(result, FrameClass::yawSmall, yaw > 0 && yaw < SMALL_ANGLE_TAN);
    appendIf(result, FrameClass::yawMedium, yaw >= SMALL_ANGLE_TAN && yaw < LARGE_ANGLE_TAN);
    appendIf(result, FrameClass::yawLarge, yaw >= LARGE_ANGLE_TAN);

    appendIf(result, FrameClass::rollSmallN, -roll > 0 && -roll < SMALL_ANGLE_TAN);
    appendIf(result, FrameClass::rollMediumN, -roll >= SMALL_ANGLE_TAN && -roll < LARGE_ANGLE_TAN);
    appendIf(result, FrameClass::rollLargeN, -roll >= LARGE_ANGLE_TAN);

    appendIf(result, FrameClass::pitchSmallN, -pitch > 0 && -pitch < SMALL_ANGLE_TAN);
    appendIf(result, FrameClass::pitchMediumN, -pitch >= SMALL_ANGLE_TAN && -pitch < LARGE_ANGLE_TAN);
    appendIf(result, FrameClass::pitchLargeN, -pitch >= LARGE_ANGLE_TAN);

    appendIf(result, FrameClass::yawSmallN, -yaw > 0 && -yaw < SMALL_ANGLE_TAN);
    appendIf(result, FrameClass::yawMediumN, -yaw >= SMALL_ANGLE_TAN && -yaw < LARGE_ANGLE_TAN);
    appendIf(result, FrameClass::yawLargeN, -yaw >= LARGE_ANGLE_TAN);

    return result;
}

void CalibrateFrameCollector::addFrame(const std::vector<cv::Point3d> &imageGrid, const std::vector<cv::Point3d> &objectGrid, size_t w, size_t h, double cost) {
    Frame frame = {imageGrid, objectGrid, w, h, cost};
    const auto &frameClasses = getClasses(frame);

    for (auto cls : frameClasses) {
        const auto &item = map.find(cls);
        if (item == map.end()) {
            map.insert({cls, std::multiset<Frame, FrameCompare>({frame})});
        } else {
            auto &set = item->second;

            set.insert(frame);
            while (set.size() > FRAMES_PER_CLASS) {
                set.erase(set.begin());
            }
        }
    }
}

double CalibrateFrameCollector::getProgress() const {
    size_t maxTotal = (COUNT - 1) * FRAMES_PER_CLASS;
    size_t total = 0;
    for (const auto &kv : map) {
        total += kv.second.size();
    }

    return (double)total / (double)maxTotal;
}

std::vector<std::vector<cv::Point3d>> CalibrateFrameCollector::getCollectedImageGrids() const {
    std::vector<std::vector<cv::Point3d>> result;

    for (const auto &kv : map) {
        for (const auto &frame : kv.second) {
            result.push_back(frame.imageGrid);
        }
    }

    return result;
}

std::vector<std::vector<cv::Point3d>> CalibrateFrameCollector::getCollectedObjectGrids() const {
    std::vector<std::vector<cv::Point3d>> result;

    for (const auto &kv : map) {
        for (const auto &frame : kv.second) {
            result.push_back(frame.objectGrid);
        }
    }

    return result;
}