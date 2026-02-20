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


void CalibrateFrameCollector::addFrameTo(GridPreferredSizeProvider &gridPreferredSizeProvider, std::unordered_map<int, FrameRef> *m, const CalibrateFrameCollector::FrameRef &frameRef) {
    int cls = frameRef->cls;
    const auto &it = m->find(cls);

    const auto w = frameRef->w;
    const auto h = frameRef->h;

    auto preferGridSize = gridPreferredSizeProvider.getGridPreferredSize();

    const auto pw = preferGridSize == nullptr ? w : preferGridSize->w;
    const auto ph = preferGridSize == nullptr ? h : preferGridSize->h;
    const auto ew = it == m->end() ? pw : it->second->w;
    const auto eh = it == m->end() ? ph : it->second->h;

    auto key = (int) w * 255 + (int) h;

    if (it == m->end()) {
        gridPreferredSizeProvider.registerFrameStat(w, h);
        m->insert({cls, frameRef});
    // если сетка лучше или соответствует более популярному размеру
    } else if (frameRef->cost < it->second->cost || (pw == w && ph == h && (ew != w || eh != h))) {
        auto key2 = (int) ew * 255 + (int) eh;
        gridPreferredSizeProvider.replaceFrameStat(w, h, ew, eh);
        m->emplace(cls, frameRef);
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
void CalibrateFrameCollector::addFrame(GridPreferredSizeProvider &gridPreferredSizeProvider, const std::shared_ptr<Frame> &frameRef) {
    addFrameTo(gridPreferredSizeProvider, &map, frameRef);
}

void CalibrateFrameCollector::addMulticamFrames(const std::vector<std::vector<CalibrateFrameCollector::FrameRef>> &_frameSets) {
    for (const auto &f : _frameSets) {
        if (f[0] == nullptr) {
            continue;
        }
        int cls = f[0]->cls;
        double cost = f[0]->cost;
        const auto &it = this->frameSets.find(cls);

        if (it == this->frameSets.end()) {
            this->frameSets.insert({cls, f});
        } else if (cost < it->second[0]->cost) {
            this->frameSets.emplace(cls, f);
        }
    }
}

std::vector<CalibrateFrameCollector::FrameRef> CalibrateFrameCollector::getFramesSample(int n, size_t w, size_t h, bool validate) const {
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
                    if (frame->w != w || frame->h != h) {
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
    auto sz = CLASSES_CUBE_SIZE;
    auto sz2 = sz / 2;
    auto dim1 = (int)std::ceil(std::max(0.6, maxRotValue[0]) * 2 * sz2);
    auto dim2 = (int)std::ceil(std::max(0.6, maxRotValue[1]) * 2 * sz2);
    auto dim3 = (int)std::ceil(std::max(1., (maxDistValue - minDistValue)) * sz);

    cv::Mat progressView = cv::Mat::zeros(dim2 + 1, dim1 + 1, CV_8U);

    auto zStep = (unsigned char)(255 / dim3);

    for (const auto &item : map) {
        auto i = item.first;
        auto xy = i / sz;
        auto x = std::clamp(xy / sz - sz2 + (int)std::round((double)dim1 / 2), 0, dim1);
        auto y = std::clamp(xy % sz - sz2 + (int)std::round((double)dim2 / 2), 0, dim2);

        progressView.at<unsigned char>(y, x) += zStep;
    }

    cv::resize(progressView, progressView, cv::Size(640, (int)(640 * dim1 / dim2)), cv::INTER_NEAREST);
    cv::imshow("progress", progressView);

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

        addMulticamFrames({set});
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
                                           (double) frame["cost"],
                                           (double) frame["ts"]);
    frameRef->validate = (bool)((int)frame["validate"]);

    return frameRef;
}

cv::FileStorage& operator << (cv::FileStorage& fs, const std::shared_ptr<CalibrateFrameCollector::Frame> &frame) {
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
//    fs << "frames" << "[";
//    for (const auto &item: map) {
//        fs << item.second;
//    }
//
//    fs << "]";
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
