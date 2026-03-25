// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

#include <3d.hpp>
#include "CalibrateFrameCollector.h"
#include <ranges>
#include <random>

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
cv::Point3d CalibrateFrameCollector::getRotationClass(const CalibrateFrameCollector::Frame &frame) const {
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

    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);  // Формула Родригеса
    cv::Vec3d n = rotationMatrix.col(2); // нормаль доски в системе камеры

    auto x = n[0];
    auto y = n[1];
    auto z = n[2];

    return {x, y, z};
}

/**
 * Определяет список классов, которым принадлежит кадр: центр, левый верхний угол, сильный крен и т.п. CalibrateFrameCollector::FrameClass::*
 *
 * @param frame
 * @return
 */
cv::Point3d CalibrateFrameCollector::getPositionClass(const CalibrateFrameCollector::Frame &frame) const {
    int w = frameSize.width;
    int h = frameSize.height;
    const auto gw = frame.w;
    const auto gh = frame.h;

    const auto &g = frame.imageGrid;
    cv::Point2d topLeft = {g[0].x, g[0].y};
    cv::Point2d topRight = {g[gw - 1].x, g[gw - 1].y};
    cv::Point2d bottomLeft = {g[(gh - 1) * gw + 0].x, g[(gh - 1) * gw + 0].y};
    cv::Point2d bottomRight = {g[(gh - 1) * gw + gw - 1].x, g[(gh - 1) * gw + gw - 1].y};

    const double rect[] = {
            std::min(topLeft.x, bottomLeft.x),
            std::min(topLeft.y, topRight.y),
            std::max(topRight.x, bottomRight.x),
            std::max(bottomLeft.y, bottomRight.y),
    };

    auto rw = rect[2] - rect[0];
    auto rh = rect[3] - rect[1];

    auto x = rect[0] / (w - rw);
    auto y = rect[1] / (h - rh);
    auto z = 1 - sqrt(rw * rh / (w * h));

    return {x, y, z};
}

int CalibrateFrameCollector::getClass(cv::Point3d p, Dim dimX, Dim dimY, Dim dimZ) {
    return (int)(
          std::floor(std::clamp((p.x * dimX.scale + dimX.bias), 0., 1.) * dimX.size)
        + std::floor(std::clamp((p.y * dimY.scale + dimY.bias), 0., 1.) * dimX.size * dimY.size)
        + std::floor(std::clamp((p.z * dimZ.scale + dimZ.bias), 0., 1.) * dimX.size * dimY.size * dimZ.size)
    );
}

CalibrateFrameCollector::FrameRef CalibrateFrameCollector::createFrame(const std::vector<cv::Point3d> &imageGrid,
                                                                                     const std::vector<cv::Point3d> &objectGrid, size_t w, size_t h, double cost, long ts, bool validate) {
    auto pFrame = new Frame({{0, 0, 0}, {0, 0, 0}, 0, 0, imageGrid, objectGrid, w, h, cost, ts, validate});
    pFrame->rotation = getRotationClass(*pFrame);
    pFrame->position = getPositionClass(*pFrame);
    pFrame->rotationClass = getClass(pFrame->rotation, R_DIM_X, R_DIM_Y, R_DIM_Z);
    pFrame->positionClass = getClass(pFrame->position, P_DIM_X, P_DIM_Y, P_DIM_Z);

    auto frameRef = FrameRef(pFrame);

    return frameRef;
}


bool CalibrateFrameCollector::addFrameTo(GridPreferredSizeProvider &gridPreferredSizeProvider, int cls, std::unordered_map<int, FrameRef> *map, const CalibrateFrameCollector::FrameRef &frameRef) {
    const auto &it = map->find(cls);

    const auto w = frameRef->w;
    const auto h = frameRef->h;

    auto [pw, ph] = gridPreferredSizeProvider.getGridPreferredSize();
    pw = pw == 0 ? w : pw;
    ph = ph == 0 ? h : ph;
    const auto ew = it == map->end() ? pw : it->second->w;
    const auto eh = it == map->end() ? ph : it->second->h;

    if (it == map->end()) {
        gridPreferredSizeProvider.insertFrameStat(w, h);
        map->insert({cls, frameRef});
        return true;
    // если сетка лучше или соответствует более популярному размеру
    } else if (frameRef->cost < it->second->cost) {
        gridPreferredSizeProvider.replaceFrameStat(w, h, ew, eh);
        map->insert_or_assign(cls, frameRef);
        return true;
    }

    return false;
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
bool CalibrateFrameCollector::addFrame(GridPreferredSizeProvider &gridPreferredSizeProvider, const FrameRef &frameRef) {
    auto rot = addFrameTo(gridPreferredSizeProvider, frameRef->rotationClass, &rotationMap, frameRef);
    auto pos = addFrameTo(gridPreferredSizeProvider, frameRef->positionClass, &positionMap, frameRef);

    return rot || pos;
}

bool CalibrateFrameCollector::addMulticamFrameSet(const std::vector<CalibrateFrameCollector::FrameRef> &f) {
    if (f[0] == nullptr) {
        return false;
    }
    int rotationClasses = R_DIM_X.size * R_DIM_Y.size * R_DIM_Z.size;
    int cls = f[0]->rotationClass + f[0]->positionClass * rotationClasses;
    double cost = f[0]->cost;
    const auto &it = this->frameSets.find(cls);

    if (it == this->frameSets.end()) {
        this->frameSets.insert({cls, f});
        return true;
    } else if (cost < it->second[0]->cost) {
        this->frameSets.insert_or_assign(cls, f);
        return true;
    }

    return false;
}

std::vector<CalibrateFrameCollector::FrameRef> CalibrateFrameCollector::getFramesSampleFrom(int n, size_t w, size_t h, bool validate, const std::unordered_map<int, FrameRef> &map) const {
    std::vector<FrameRef> f(n);

    std::uniform_int_distribution rng(0, n - 1);
    std::mt19937 r {std::random_device{}()};

    int i = 0;
    for (const auto &frame : map) {
        if (frame.second->validate == validate && frame.second->w == w && frame.second->h == h) {
            if (i < n) {
                f[i] = frame.second;
            } else {
                f[rng(r) % n] = frame.second;
            }
            i++;
        }
    }

    if (i < n) {
        f.resize(i);
    }

    return f;
}

std::vector<CalibrateFrameCollector::FrameRef> CalibrateFrameCollector::getFramesSample(int n, size_t w, size_t h, bool validate) const {
    if (rotationMap.empty() && positionMap.empty()) {
        return {};
    }
    if (rotationMap.empty()) {
        return getFramesSampleFrom(n, w, h, validate, positionMap);
    }
    if (positionMap.empty()) {
        return getFramesSampleFrom(n, w, h, validate, rotationMap);
    }

    int nR = n * (int)positionMap.size() / (int)(positionMap.size() + rotationMap.size());
    int nP = n - nR;
    std::vector<FrameRef> fR, fP;
    fR.reserve(n);
    fP.reserve(nP);

    fR = getFramesSampleFrom(nR, w, h, validate, rotationMap);
    fP = getFramesSampleFrom(nP, w, h, validate, positionMap);

    fR.insert(fR.end(), fP.begin(), fP.end());

    return fR;
}

std::vector<std::vector<CalibrateFrameCollector::FrameRef>> CalibrateFrameCollector::getFrameSetsSample(int n, size_t w, size_t h, bool validate) const {
    std::vector<std::vector<FrameRef>> f(n);

    std::uniform_int_distribution rng(0, n - 1);
    std::mt19937 r {std::random_device{}()};

    int i = 0;
    for (const auto &frameSet : frameSets) {
        if (frameSet.second[0]->validate == validate) {
            if (w > 0) {
                auto valid = true;
                for (const auto &frame: frameSet.second) {
                    if (frame == nullptr || frame->w != w || frame->h != h) {
                        valid = false;
                    }
                }
                if (!valid) {
                    continue;
                }
            }

            if (i < n) {
                f[i] = frameSet.second;
            } else {
                f[rng(r) % n] = frameSet.second;
            }
            i++;
        }
    }

    if (i < n) {
        f.resize(i);
    }

    return f;
}

double CalibrateFrameCollector::getProgress() const {
//#ifdef HAVE_OPENCV_HIGHGUI
//    cv::Mat RView = cv::Mat::zeros(P_DIM_Y.size, P_DIM_X.size, CV_8U);
//    cv::Mat PView = cv::Mat::zeros(P_DIM_Y.size, P_DIM_X.size, CV_8U);
//
//    auto zStep = (unsigned char)(255 / P_DIM_Z.size);
//
//    for (const auto &item : positionMap) {
//        PView.at<unsigned char>((int)item.second->position.x, (int)item.second->position.y) += zStep;
//    }
//
//    cv::imshow("progress P", PView);
//
//    zStep = (unsigned char)(255 / R_DIM_Z.size);
//
//    for (const auto &item : rotationMap) {
//        RView.at<unsigned char>((int)item.second->position.x, (int)item.second->position.y) += zStep;
//    }
//
//    cv::resize(RView, RView, cv::Size(640, (int)(640 * P_DIM_X.size / P_DIM_Y.size)), cv::INTER_NEAREST);
//    cv::resize(PView, PView, cv::Size(640, (int)(640 * P_DIM_X.size / P_DIM_Y.size)), cv::INTER_NEAREST);
//    cv::imshow("progress R", RView);
//#endif
    return (double)(positionMap.size() + rotationMap.size()) / (double)TOTAL_VOLUME;
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

void CalibrateFrameCollector::load(GridPreferredSizeProvider &gridPreferredSizeProvider, const cv::FileStorage &fs) {
    cv::FileNode framesNode = fs["frames"];

    for (const auto &frame: framesNode) {
        if (!frame.isMap()) {
            continue;
        }
        const auto &frameRef = loadFrame(frame);

        if (frameRef == nullptr) {
            continue;
        }

        addFrame(gridPreferredSizeProvider, frameRef);
    }
    cv::FileNode framesPairsNode = fs["multicam"];

    for (const auto &frameSet: framesPairsNode) {
        if (!frameSet.isSeq()) {
            continue;
        }

        std::vector<FrameRef> set;
        for (const auto &frame : frameSet) {
            set.emplace_back(loadFrame(frame));
        }

        addMulticamFrameSet(set);
    }
}

CalibrateFrameCollector::FrameRef CalibrateFrameCollector::loadFrame(const cv::FileNode &frame) {
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
                                           (double) frame["cost"],
                                           (double) frame["ts"],
                                           (bool)((int)frame["validate"]));
    return frameRef;
}

cv::FileStorage& operator << (cv::FileStorage& fs, const CalibrateFrameCollector::FrameRef &frame) {
    fs << "{" << "w" << (int) frame->w << "h" << (int) frame->h << "cost" << frame->cost << "ts" << frame->ts << "validate" << frame->validate;
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
    std::set<double> processed;
    fs << "frames" << "[";
    for (const auto &item: rotationMap) {
        processed.insert(item.second->ts);
        fs << item.second;
    }
    for (const auto &item: positionMap) {
        if (processed.find(item.second->ts) == processed.end()) {
            fs << item.second;
        }
    }
    fs << "]";
    fs << "multicam" << "[";
    for (const auto &frameSet: frameSets) {
        fs << "[";
        for (const auto &frame : frameSet.second) {
            fs << frame;
        }
        fs << "]";
    }

    fs << "]";
}

size_t CalibrateFrameCollector::getFrameCount() const {
    return rotationMap.size() + positionMap.size();
}

size_t CalibrateFrameCollector::getFrameSetCount() const {
    return frameSets.size();
}

void CalibrateFrameCollector::reset() {
    frameSets.clear();
    rotationMap.clear();
    positionMap.clear();
}
