#include <imgproc.hpp>
#include <iostream>
#include "DisparityEvaluator.h"

namespace ecv {
    int16_t getDisparity(const uint8_t *data1, const uint8_t *data2, size_t x, size_t y, int16_t minDisparity, int16_t maxDisparity, size_t windowSize, int8_t sz);

    void DisparityEvaluator::evaluateDisparity(const std::vector<cv::Mat> &frames, cv::Mat &disparity) const {
        for (int i = 1; i < frames.size(); i++) {
            CV_Assert(frames[0].cols == frames[i].cols);
            CV_Assert(frames[0].rows == frames[i].rows);
            CV_Assert(frames[0].channels() == frames[i].channels());
            CV_Assert(frames[0].type() == frames[i].type());
            CV_Assert(frames[0].elemSize() == frames[i].elemSize());
        }

        if (disparity.empty()) {
            disparity = cv::Mat(frames[0].size(), CV_16S);
        } else {
            CV_Assert(disparity.cols == frames[0].cols);
            CV_Assert(disparity.rows == frames[0].rows);
            CV_Assert(disparity.type() == CV_16S);
        }

        /** last image in piramid should not less 100px width */
        auto pyramidSize = (size_t)std::max(0., std::floor(std::log(frames[0].cols / 300.0) / std::log(2)));
        std::vector piramids(pyramidSize, std::vector<cv::Mat> (frames.size()));
        std::vector<cv::Mat> roughDisparity(pyramidSize + 1);

        for (int j = 0; j < frames.size(); j++) {
            cv::resize(frames[j], piramids[0][j], cv::Size(frames[j].cols / 2, frames[j].rows / 2));
        }

        for (int i = 1; i < pyramidSize; ++i) {
            for (int j = 0; j < frames.size(); j++) {
                cv::resize(piramids[i - 1][j], piramids[i][j], cv::Size(piramids[i - 1][j].cols / 2, piramids[i - 1][j].rows / 2));
            }
        }

        for (size_t i = pyramidSize; i > 0; i--) {
            this->evaluateIncrementally(piramids[i - 1], roughDisparity[i], roughDisparity[i - 1]);
        }

        this->evaluateIncrementally(frames, roughDisparity[0], disparity);
    }

    void DisparityEvaluator::evaluateIncrementally(const std::vector<cv::Mat> &frames, const cv::Mat &roughDisparity, cv::Mat &disparity) const {
        for (int i = 1; i < frames.size(); i++) {
            CV_Assert(frames[0].cols == frames[i].cols);
            CV_Assert(frames[0].rows == frames[i].rows);
            CV_Assert(frames[0].channels() == frames[i].channels());
            CV_Assert(frames[0].type() == frames[i].type());
            CV_Assert(frames[0].elemSize() == frames[i].elemSize());
        }

        if (disparity.empty()) {
            disparity = cv::Mat(frames[0].size(), CV_16S);
            disparity.setTo(0);
        } else {
            CV_Assert(disparity.cols == frames[0].cols);
            CV_Assert(disparity.rows == frames[0].rows);
            CV_Assert(disparity.type() == CV_16S);
        }

        auto sz = (int8_t)frames[0].elemSize();
        auto w = frames[0].cols;
        auto wsz = sz * w;
        auto maxDisparity = (int16_t)(48 * sz);
        auto windowSize = (size_t)64 * sz;

        std::vector<uint8_t *> data(frames.size());

        for (int i = 0; i < frames.size(); ++i) {
            data[i] = frames[i].data;
        }

        auto disparityData = (int16_t *)disparity.data;
        auto roughDisparityData = (int16_t *)roughDisparity.data;
        auto rw = roughDisparity.cols;

        if (!roughDisparity.empty()) {
            CV_Assert(roughDisparity.cols == frames[0].cols / 2);
            CV_Assert(roughDisparity.rows == frames[0].rows / 2);
            CV_Assert(disparity.type() == CV_16S);
            maxDisparity = (int16_t)(8 * sz);
            windowSize = 32 * sz;
        }

        if (roughDisparity.empty()) {
#pragma omp parallel for
            for (size_t y = 0; y < frames[0].rows; y++) {
                auto yptr = y * wsz;
                for (size_t x = 0; x < maxDisparity; x += sz) {
                    auto d = getDisparity(data[0], data[1], x, yptr, (int16_t)-x, maxDisparity, windowSize, sz);
                    disparityData[y * w + x / sz] = std::abs(d - disparityData[y * w + x / sz - 1]) == 1 ? disparityData[y * w + x / sz - 1] : d;
                }
                for (size_t x = maxDisparity; x < wsz - windowSize - maxDisparity; x += sz) {
                    auto d = getDisparity(data[0], data[1], x, yptr, (int16_t)-maxDisparity,  maxDisparity, windowSize, sz);
                    disparityData[y * w + x / sz] = std::abs(d - disparityData[y * w + x / sz - 1]) == 1 ? disparityData[y * w + x / sz - 1] : d;
                }
                for (size_t x = wsz - windowSize - maxDisparity; x < wsz - windowSize; x += sz) {
                    auto d = getDisparity(data[0], data[1], x, yptr, (int16_t)-maxDisparity, (int16_t) (wsz - windowSize - x), windowSize, sz);
                    disparityData[y * w + x / sz] = std::abs(d - disparityData[y * w + x / sz - 1]) == 1 ? disparityData[y * w + x / sz - 1] : d;
                }
            }
        } else {
#pragma omp parallel for
            for (size_t y = 0; y < frames[0].rows; y++) {
                auto yptr = y * wsz;
                auto syptr = y / 2 * rw;
                for (size_t x = 0; x < maxDisparity; x += sz) {
                    auto suggest = (int16_t)(roughDisparityData[syptr + x / sz / 2] * 2);
                    auto d = getDisparity(data[0], data[1], x, yptr, (int16_t)-x, (int16_t) (suggest + maxDisparity), windowSize, sz);
                    disparityData[y * w + x / sz] = std::abs(d - disparityData[y * w + x / sz - 1]) == 1 ? disparityData[y * w + x / sz - 1] : d;
                }
                for (size_t x = maxDisparity; x < wsz - windowSize - maxDisparity; x += sz) {
                    auto suggest = (int16_t)(roughDisparityData[syptr + x / sz / 2] * 2);
                    auto d = getDisparity(data[0], data[1], x, yptr, (int16_t) (suggest - maxDisparity), (int16_t) (suggest + maxDisparity), windowSize, sz);
                    disparityData[y * w + x / sz] = std::abs(d - disparityData[y * w + x / sz - 1]) == 1 ? disparityData[y * w + x / sz - 1] : d;
                }
                for (size_t x = wsz - windowSize - maxDisparity; x < wsz - windowSize; x += sz) {
                    auto suggest = (int16_t)(roughDisparityData[syptr + x / sz / 2] * 2);
                    auto d = getDisparity(data[0], data[1], x, yptr, (int16_t) (suggest - maxDisparity), (int16_t) (wsz - windowSize - x), windowSize, sz);
                    disparityData[y * w + x / sz] = std::abs(d - disparityData[y * w + x / sz - 1]) == 1 ? disparityData[y * w + x / sz - 1] : d;
                }
            }
        }
    }

    int eval(const uint8_t *src, const uint8_t *dest, size_t windowSize, int8_t sz);

    inline int16_t getDisparity(const uint8_t *data1, const uint8_t *data2, size_t x, size_t y, int16_t minDisparity, int16_t maxDisparity, size_t windowSize, int8_t sz) {
        const uint8_t *src = data1 + y + x;
        const uint8_t *dest = data2 + y + x;
        auto score = std::numeric_limits<int>::max();
        int16_t disparity = 0;

        for (int i = minDisparity; i < maxDisparity; i += sz) {
            auto newScore = eval(src, dest + i, windowSize, sz);
            newScore += eval(dest, src - i, windowSize, sz);
            if (newScore < score) {
                score = newScore;
                disparity = (int16_t)i;
            }
        }

        return -1*disparity;
    }

    inline int eval(const uint8_t *src, const uint8_t *dest, size_t windowSize, int8_t sz) {
        auto score = 0;
        for (int i = 0; i < windowSize; ++i) {
            auto d = std::abs(((int)src[i] - (int)src[i + sz]) - ((int)dest[i] - (int)dest[i + sz]));
            score += d;
        }

        return score;
    }
} // ecv