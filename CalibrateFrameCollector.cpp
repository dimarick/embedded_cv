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

/**
 * Определяет список классов, которым принадлежит кадр: центр, левый верхний угол, сильный крен и т.п. CalibrateFrameCollector::FrameClass::*
 *
 * @param frame
 * @return
 */
std::vector<CalibrateFrameCollector::FrameClass> CalibrateFrameCollector::getClasses(const CalibrateFrameCollector::Frame &frame) const {
    int w = frameSize.width;
    int h = frameSize.height;
    int w2 = w / 2;
    int h2 = h / 2;
    int w4 = w / 4;
    int h4 = h / 4;
    int w8 = w / 8;
    int h8 = h / 8;
    int w16 = w / 16;
    int h16 = h / 16;
    int w2_16 = w2 - w16;
    int h2_16 = h2 - h16;
    int w_8 = w - w8;
    int h_8 = h - h8;
    auto rTopLeft =     cv::Rect(w16, w16, w2_16, w2_16);
    auto rTopRight =    cv::Rect(w2,  w16, w2_16, h2_16);
    auto rBottomLeft =  cv::Rect(w16, h2,  w2_16, h2_16);
    auto rBottomRight = cv::Rect(w2,  h2,  w2_16, h2_16);

    auto rCenter = cv::Rect(w4, h4, w2, h2);
    auto rAll =    cv::Rect(w16, h16, w_8, h_8);

    const auto &g = frame.imageGrid;
    const auto gw = frame.w;
    const auto gh = frame.h;

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
            100000, 0, w2, // fx fy заведомо запредельные,
            0, 100000, h2, // так стабильнее оценивается угол по искажениям независимо от реального фокусного и расстояния до сетки
            0, 0, 1);

    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);  // Без дисторсии
    std::vector<double> rvec({0, 0, 0}), tvec({0, 0, 0});

    std::vector<cv::Point3d> objectPoints = {
            {0, 0, 0},
            {(double)gw - 1, 0, 0},
            {0, (double)gh - 1, 0},
            {(double)gw - 1, (double)gh - 1, 0},
    };
    std::vector<cv::Point2d> imagePoints = {
            {g[0].x, g[0].y},
            {g[gw - 1].x, g[gw - 1].y},
            {g[(gh - 1) * gw + 0].x, g[(gh - 1) * gw + 0].y},
            {g[(gh - 1) * gw + gw - 1].x, g[(gh - 1) * gw + gw - 1].y},
    };

    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);  // Формула Родригеса

    // 5. Извлекаем углы Эйлера из матрицы вращения
    // Порядок вращений: Z (yaw), Y (pitch), X (roll)
    double sy = sqrt(rotationMatrix.at<double>(0,0) * rotationMatrix.at<double>(0,0) +
                     rotationMatrix.at<double>(1,0) * rotationMatrix.at<double>(1,0));

    bool singular = sy < 1e-6;

    yaw = atan2(-rotationMatrix.at<double>(2,0), sy);
    if (!singular) {
        pitch = atan2(rotationMatrix.at<double>(2,1),
                            rotationMatrix.at<double>(2,2));
        roll = atan2(rotationMatrix.at<double>(1,0),
                           rotationMatrix.at<double>(0,0));
    } else {
        pitch = atan2(-rotationMatrix.at<double>(1,2),
                            rotationMatrix.at<double>(1,1));
        roll = 0;
    }

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

    appendIf(result, FrameClass::rollSmall, roll > 0 && roll < SMALL_ANGLE * 0.5);
    appendIf(result, FrameClass::rollMedium, roll >= SMALL_ANGLE * 0.5 && roll < LARGE_ANGLE * 0.5);
    appendIf(result, FrameClass::rollLarge, roll >= LARGE_ANGLE * 0.5);

    appendIf(result, FrameClass::pitchSmall, pitch > 0 && pitch < SMALL_ANGLE);
    appendIf(result, FrameClass::pitchMedium, pitch >= SMALL_ANGLE && pitch < LARGE_ANGLE);
    appendIf(result, FrameClass::pitchLarge, pitch >= LARGE_ANGLE);

    appendIf(result, FrameClass::yawSmall, yaw > 0 && yaw < SMALL_ANGLE);
    appendIf(result, FrameClass::yawMedium, yaw >= SMALL_ANGLE && yaw < LARGE_ANGLE);
    appendIf(result, FrameClass::yawLarge, yaw >= LARGE_ANGLE);

    appendIf(result, FrameClass::rollSmallN, -roll > 0 && -roll < SMALL_ANGLE * 0.5);
    appendIf(result, FrameClass::rollMediumN, -roll >= SMALL_ANGLE * 0.5 && -roll < LARGE_ANGLE * 0.5);
    appendIf(result, FrameClass::rollLargeN, -roll >= LARGE_ANGLE * 0.5);

    appendIf(result, FrameClass::pitchSmallN, -pitch > 0 && -pitch < SMALL_ANGLE);
    appendIf(result, FrameClass::pitchMediumN, -pitch >= SMALL_ANGLE && -pitch < LARGE_ANGLE);
    appendIf(result, FrameClass::pitchLargeN, -pitch >= LARGE_ANGLE);

    appendIf(result, FrameClass::yawSmallN, -yaw > 0 && -yaw < SMALL_ANGLE);
    appendIf(result, FrameClass::yawMediumN, -yaw >= SMALL_ANGLE && -yaw < LARGE_ANGLE);
    appendIf(result, FrameClass::yawLargeN, -yaw >= LARGE_ANGLE);

    return result;
}

/**
 * Регистрирует кадр в двух хранилищах: с классификацией по классам (map) и без (frames с контролем уникальности).
 * Каждый кадр может присутствовать не более FRAMES_PER_CLASS раз в каждом класса.
 * Если кадров больше - выбираются наиболее удачные FRAMES_PER_CLASS кадров.
 * Заполнение FrameClass::COUNT * FRAMES_PER_CLASS кадров означает полное завершение калибровки
 *
 * @param imageGrid
 * @param objectGrid
 * @param w
 * @param h
 * @param cost
 */
void CalibrateFrameCollector::addFrame(const std::vector<cv::Point3d> &imageGrid, const std::vector<cv::Point3d> &objectGrid, size_t w, size_t h, double cost) {
    auto frameRef = std::shared_ptr<Frame>(new Frame({imageGrid, objectGrid, w, h, cost}));
    auto result = frames.insert({frameRef, 0});
    auto frame = result.first->first.get();

    const auto &frameClasses = getClasses(*frame);

    for (auto cls : frameClasses) {
        const auto &item = map.find(cls);
        if (item == map.end()) {
            map.insert({cls, std::multiset<std::shared_ptr<Frame>, FrameCompare>({frameRef})});
            result.first->second++;
        } else {
            auto &set = item->second;
            set.insert({frameRef});
            result.first->second++;
        }
    }

    for (auto cls : frameClasses) {
        const auto &item = map.find(cls);
        if (item != map.end()) {
            auto &set = item->second;
            while (set.size() > FRAMES_PER_CLASS) {
                const auto &it = set.begin();
                const auto &refToDelete = *it;
                set.erase(it);
                const auto &frameIt = frames.find(refToDelete);
                if (frameIt != frames.end()) {
                    frameIt->second--;
                    if (frameIt->second <= 0) {
                        frames.erase(frameIt);
                    }
                }
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

    std::vector<const char *> missing;

    appendIf(missing, "topLeft", map.find(topLeft) == map.end() || map.find(topLeft)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "topRight", map.find(topRight) == map.end() || map.find(topRight)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "bottomLeft", map.find(bottomLeft) == map.end() || map.find(bottomLeft)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "bottomRight", map.find(bottomRight) == map.end() || map.find(bottomRight)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "center", map.find(center) == map.end() || map.find(center)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "near", map.find(near) == map.end() || map.find(near)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "mid", map.find(mid) == map.end() || map.find(mid)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "far", map.find(far) == map.end() || map.find(far)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "rollSmall", map.find(rollSmall) == map.end() || map.find(rollSmall)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "yawSmall", map.find(yawSmall) == map.end() || map.find(yawSmall)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "pitchSmall", map.find(pitchSmall) == map.end() || map.find(pitchSmall)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "rollMedium", map.find(rollMedium) == map.end() || map.find(rollMedium)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "yawMedium", map.find(yawMedium) == map.end() || map.find(yawMedium)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "pitchMedium", map.find(pitchMedium) == map.end() || map.find(pitchMedium)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "rollLarge", map.find(rollLarge) == map.end() || map.find(rollLarge)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "yawLarge", map.find(yawLarge) == map.end() || map.find(yawLarge)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "pitchLarge", map.find(pitchLarge) == map.end() || map.find(pitchLarge)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "rollSmallN", map.find(rollSmallN) == map.end() || map.find(rollSmallN)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "yawSmallN", map.find(yawSmallN) == map.end() || map.find(yawSmallN)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "pitchSmallN", map.find(pitchSmallN) == map.end() || map.find(pitchSmallN)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "rollMediumN", map.find(rollMediumN) == map.end() || map.find(rollMediumN)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "yawMediumN", map.find(yawMediumN) == map.end() || map.find(yawMediumN)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "pitchMediumN", map.find(pitchMediumN) == map.end() || map.find(pitchMediumN)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "rollLargeN", map.find(rollLargeN) == map.end() || map.find(rollLargeN)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "yawLargeN", map.find(yawLargeN) == map.end() || map.find(yawLargeN)->second.size() < FRAMES_PER_CLASS);
    appendIf(missing, "pitchLargeN", map.find(pitchLargeN) == map.end() || map.find(pitchLargeN)->second.size() < FRAMES_PER_CLASS);

    std::cout << std:: format("Missing: {}\n", missing) << std::endl;

    return (double)total / (double)maxTotal;
}

std::vector<std::vector<cv::Point3d>> CalibrateFrameCollector::getCollectedImageGrids() const {
    std::vector<std::vector<cv::Point3d>> result;

    for (const auto &kv : map) {
        for (const auto &frame : kv.second) {
            result.push_back(frame->imageGrid);
        }
    }

    return result;
}

std::vector<std::vector<cv::Point3d>> CalibrateFrameCollector::getCollectedObjectGrids() const {
    std::vector<std::vector<cv::Point3d>> result;

    for (const auto &kv : map) {
        for (const auto &frame : kv.second) {
            result.push_back(frame->objectGrid);
        }
    }

    return result;
}

void CalibrateFrameCollector::load(const cv::FileStorage &fs) {
    cv::FileNode framesNode = fs["frames"];

    for (const auto &frame: framesNode) {
        std::vector<cv::Point3d> imagePoints, objectPoints;
        if (frame["imagePoints"].size() != frame["objectPoints"].size()) {
            continue;
        }

        for (const auto &point: frame["imagePoints"]) {
            imagePoints.emplace_back(point["x"], point["y"], 0);
        }
        for (const auto &point: frame["objectPoints"]) {
            objectPoints.emplace_back(point["x"], point["y"], 0);
        }
        addFrame(imagePoints, objectPoints, (int)frame["w"], (int)frame["h"], (double)frame["cost"]);
    }
}

void CalibrateFrameCollector::store(cv::FileStorage &fs) {
    fs << "frames" << "[";
    for (const auto &item: getFrames()) {
        const auto frame  = item.first;
        fs << "{" << "w" << (int) frame->w << "h" << (int) frame->h << "cost"
           << frame->cost;
        std::vector<cv::Point3d> imagePoints, objectPoints;
        fs << "imagePoints" << "[";
        for (const auto &point: frame->imageGrid) {
            fs << "{" << "x" << point.x << "y" << point.y << "}";
        }
        fs << "]";
        fs << "objectPoints" << "[";
        for (const auto &point: frame->objectGrid) {
            fs << "{" << "x" << point.x << "y" << point.y << "}";
        }
        fs << "]" << "}";
    }

    fs << "]";
}
