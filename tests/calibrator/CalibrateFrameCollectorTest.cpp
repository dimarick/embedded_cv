#include <catch2/catch_test_macros.hpp>
#include "../../src/calibrator/CalibrateFrameCollector.h"
#include <opencv2/core.hpp>

using namespace ecv;

// Создаём фиктивный кадр с заданными координатами углов
std::shared_ptr<CalibrateFrameCollector::Frame> makeFrame(
        const std::vector<cv::Point3d>& imageGrid,
        size_t w, size_t h, double cost = 1.0) {
    std::vector<cv::Point3d> objectGrid(w*h);
    for (size_t i = 0; i < w*h; ++i)
        objectGrid[i] = cv::Point3d(i % w, i / w, 0);
    auto frame = std::make_shared<CalibrateFrameCollector::Frame>(
        CalibrateFrameCollector::Frame{{0,0,0}, {0,0,0}, 0, 0, imageGrid, objectGrid, w, h, cost, 0, false});
    return frame;
}

TEST_CASE("Frame classification", "[collector]") {
    cv::Size frameSize(1920, 1080);
    CalibrateFrameCollector collector(frameSize);

    // Случай 1: доска прямо, центр совпадает с центром кадра
    std::vector<cv::Point3d> grid(4*4);
    int w = 4, h = 4;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            grid[y*w + x] = cv::Point3d(500 + x*100, 400 + y*100, 1.0);
        }
    }
    auto frame = makeFrame(grid, w, h);
    int cls = collector.getClass(collector.getRotationClass(*frame), collector.R_DIM_X, collector.R_DIM_Y, collector.R_DIM_Z);
    // Класс должен быть близок к центру куба классов (ориентация нулевая, расстояние среднее)
    // Проверим, что cls в допустимом диапазоне (0 .. CLASSES_CUBE_SIZE^3)
    int maxClass = collector.getDatasetVolume();
    REQUIRE(cls >= 0);
    REQUIRE(cls < maxClass);
}