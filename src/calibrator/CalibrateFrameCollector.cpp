#include <3d.hpp>
#include "CalibrateFrameCollector.h"
#include <ranges>
#include <random>
#include <highgui.hpp>
#include <imgproc.hpp>

using namespace ecv;

double distance(cv::Point2d p1, cv::Point2d p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

/**
 * Определяет список классов, которым принадлежит кадр: центр, левый верхний угол, сильный крен и т.п. CalibrateFrameCollector::FrameClass::*
 *
 * @param frame
 * @return
 */
int CalibrateFrameCollector::getClass(const CalibrateFrameCollector::Frame &frame) {
    int w = frameSize.width;
    int h = frameSize.height;
    int w2 = w / 2;
    int h2 = h / 2;

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

    cv::Point2d center = {g[gh2 * gw + gw2].x, g[gh2 * gw + gw2].y};

    std::vector<cv::Point2d> sides = {
            {g[gh2 * gw + gw2 - 1].x, g[gh2 * gw + gw2 - 1].y},
            {g[gh2 * gw + gw2 + 1].x, g[gh2 * gw + gw2 + 1].y},
            {g[gh2 * gw - gw + gw2].x, g[gh2 * gw - gw + gw2].y},
            {g[gh2 * gw + gw + gw2].x, g[gh2 * gw + gw + gw2].y},
    };

    auto avgCellSize = 0.;
    for (const auto &side : sides) {
        avgCellSize += std::sqrt(std::pow(side.x - center.x, 2) + std::pow(side.y - center.y, 2)) / (double)sides.size();
    }

    auto maxPossibleCellSize = (double)std::min(w, h) / 6;

    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);  // Формула Родригеса
    cv::Vec3d n = rotationMatrix.col(2); // нормаль доски в системе камеры
    auto dist = avgCellSize / maxPossibleCellSize;

    auto sz = CLASSES_CUBE_SIZE;
    auto sz2 = ((double)sz - 1) / 2.;


    auto x = (int)(std::round((n[0] + 1) * sz2));
    auto y = (int)(std::round((n[1] + 1) * sz2));
    auto z = (int)(std::round(dist * sz));

    int offset = x * sz * sz + y * sz + z;

    std::cout << "getClass x " << n[0] << " y " << n[1] << " z " << n[2] << " d " << dist << " sin(d) " << dist << "(x,y,z)" << x << "," << y << "," << z << std::endl;

    maxRotValue[0] = std::max(maxRotValue[0], std::abs(n[0]));
    maxRotValue[1] = std::max(maxRotValue[1], std::abs(n[1]));
    maxDistValue = std::max(maxDistValue, dist);
    minDistValue = std::min(minDistValue, dist);

    return offset;
}

std::shared_ptr<CalibrateFrameCollector::Frame> CalibrateFrameCollector::createFrame(const std::vector<cv::Point3d> &imageGrid,
                                                                                     const std::vector<cv::Point3d> &objectGrid, size_t w, size_t h, double cost, double ts) {
    auto frameRef = std::shared_ptr<Frame>(new Frame({0, imageGrid, objectGrid, w, h, cost, ts, std::rand() % 2 == 0}));
    frameRef->cls = getClass(*frameRef);

    return frameRef;
}


void CalibrateFrameCollector::addFrameTo(std::unordered_map<int, FrameRef> *m, const CalibrateFrameCollector::FrameRef &frameRef) {
    int cls = frameRef->cls;
    const auto &it = m->find(cls);

    if (it == m->end()) {
        m->insert({cls, frameRef});
    } else if (frameRef->cost < it->second->cost) {
        m->emplace(cls, frameRef);
    }
}

void CalibrateFrameCollector::addMulticamFrameTo(std::unordered_map<int, FramePairRef> *m,
                                                 const CalibrateFrameCollector::FramePairRef &framePairRef) {
    int cls = framePairRef->getClass();
    const auto &it = m->find(cls);

    if (it == m->end()) {
        m->insert({cls, framePairRef});
    } else if (framePairRef->cost < it->second->cost) {
        m->emplace(cls, framePairRef);
    }
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
void CalibrateFrameCollector::addFrame(const std::shared_ptr<Frame> &frameRef) {
    addFrameTo(&map, frameRef);
}

void CalibrateFrameCollector::addMulticamFrame(const std::shared_ptr<Frame> &baseFrameRef,
                                               const std::shared_ptr<Frame> &frameRef, double cost) {
    auto pair = FramePairRef(new FramePair{cost, baseFrameRef, frameRef});

    addMulticamFrameTo(&pairs, pair);
}

std::vector<std::pair<int, CalibrateFrameCollector::FrameRef>> CalibrateFrameCollector::getFramesSample(int n) const {
    std::vector<std::pair<int, FrameRef>> f;

    std::mt19937 r {std::random_device{}()};

    std::sample(map.begin(), map.end(), std::back_inserter(f), n, r);

    return f;
}

std::vector<std::pair<int, CalibrateFrameCollector::FramePairRef>> CalibrateFrameCollector::getFramesPairsSample(int n) const {
    std::vector<std::pair<int, FramePairRef>> f;

    std::mt19937 r {std::random_device{}()};

    std::sample(pairs.begin(), pairs.end(), std::back_inserter(f), n, r);

    return f;
}

double CalibrateFrameCollector::getProgress() const {
    auto sz = CLASSES_CUBE_SIZE;
    auto sz2 = sz / 2;
    auto dim1 = (int)std::ceil(maxRotValue[0] * 2 * sz2);
    auto dim2 = (int)std::ceil(maxRotValue[1] * 2 * sz2);
    auto dim3 = (int)std::ceil((maxDistValue - minDistValue) * sz);

    cv::Mat progressView = cv::Mat::zeros(dim2 + 1, dim1 + 1, CV_8U);

    auto zStep = (unsigned char)(255 / dim3);

//    for (const auto &item : map) {
//        auto i = item.first;
//        auto xy = i / sz;
//        auto x = xy / sz - sz2 + (int)std::round((double)dim1 / 2);
//        auto y = xy % sz - sz2 + (int)std::round((double)dim2 / 2);
//
//        CV_Assert(x >= 0 && x < dim1 + 1);
//        CV_Assert(y >= 0 && y < dim2 + 1);
//
//        progressView.at<unsigned char>(y, x) += zStep;
//    }
//
//    cv::resize(progressView, progressView, cv::Size(640, (int)(640 * dim1 / dim2)), cv::INTER_NEAREST);
//    cv::imshow("progress", progressView);

    return (double)map.size() / (double)(3.14 / 4 * dim1 * dim2 * dim3);
}

std::vector<std::vector<cv::Point3d>> CalibrateFrameCollector::getCollectedImageGridsSample(const std::vector<CalibrateFrameCollector::FrameRef> &sample) const {
    std::vector<std::vector<cv::Point3d>> result;

    for (const auto &frame : sample) {
        result.push_back(frame->imageGrid);
    }

    return result;
}

std::vector<std::vector<cv::Point3d>> CalibrateFrameCollector::getCollectedObjectGridsSample(const std::vector<CalibrateFrameCollector::FrameRef> &sample) const {
    std::vector<std::vector<cv::Point3d>> result;

    for (const auto &frame : sample) {
        result.push_back(frame->objectGrid);
    }

    return result;
}

void CalibrateFrameCollector::load(const cv::FileStorage &fs) {
    cv::FileNode framesNode = fs["frames"];

    for (const auto &frame: framesNode) {
        if (!frame.isMap()) {
            continue;
        }
        const auto &frameRef = loadFrame(frame);

        if (frameRef == nullptr) {
            continue;
        }

        addFrame(frameRef);
    }
    cv::FileNode framesPairsNode = fs["multicam"];

    for (const auto &framePair: framesPairsNode) {
        if (!framePair.isMap() || !framePair["a"].isMap() || !framePair["b"].isMap()) {
            continue;
        }
        const auto &baseFrameRef = loadFrame(framePair["a"]);
        const auto &frameRef = loadFrame(framePair["b"]);

        if (baseFrameRef == nullptr || frameRef == nullptr) {
            continue;
        }

        addMulticamFrame(baseFrameRef, frameRef, (double)framePair["cost"]);
    }
}

std::shared_ptr<CalibrateFrameCollector::Frame> CalibrateFrameCollector::loadFrame(const cv::FileNode &frame) {
    std::vector<cv::Point3d> imagePoints, objectPoints;
    if (frame["imagePoints"].size() != frame["objectPoints"].size()) {
        return nullptr;
    }

    for (const auto &point: frame["imagePoints"]) {
        imagePoints.emplace_back(point["x"], point["y"], 0);
    }
    for (const auto &point: frame["objectPoints"]) {
        objectPoints.emplace_back(point["x"], point["y"], 0);
    }
    if (imagePoints.empty() || objectPoints.empty()) {
        return nullptr;
    }

    const FrameRef &frameRef = createFrame(imagePoints, objectPoints, (int) frame["w"], (int) frame["h"],
                                           (double) frame["cost"]);
    frameRef->validate = (bool)((int)frame["validate"]);

    return frameRef;
}

cv::FileStorage& operator << (cv::FileStorage& fs, const std::shared_ptr<CalibrateFrameCollector::Frame> &frame) {
    fs << "{" << "w" << (int) frame->w << "h" << (int) frame->h << "cost" << frame->cost << "validate" << frame->validate;
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

    return fs;
}

void CalibrateFrameCollector::store(cv::FileStorage &fs) const {
    fs << "frames" << "[";
    for (const auto &item: map) {
        fs << item.second;
    }

    fs << "]";
    fs << "multicam" << "[";
    for (const auto &framePair: pairs) {
        fs << "{" << "a" << framePair.second->base;
        fs << "b" << framePair.second->current;
        fs << "cost" << framePair.second->cost << "}";
    }

    fs << "]";
}
