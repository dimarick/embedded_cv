#include <imgproc.hpp>
#include <iostream>
#include "DisparityEvaluator.h"

namespace ecv {
    int16_t getDisparity(const uint8_t *data1, const uint8_t *data2, size_t x, size_t y, size_t w, size_t h, int minDisparity, int maxDisparity, size_t windowSize, uint8_t sz, bool allowFail = false);

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
        auto pyramidSize = std::min((size_t)2, (size_t)std::max(0., std::floor(std::log(frames[0].cols / 200.0) / std::log(2))));
        std::vector piramids(pyramidSize, std::vector<cv::Mat> (frames.size()));
        std::vector<cv::Mat> roughDisparity(pyramidSize + 1);

        if (pyramidSize > 0) {
            for (int j = 0; j < frames.size(); j++) {
                cv::resize(frames[j], piramids[0][j], cv::Size(frames[j].cols / 2, frames[j].rows / 2));
            }
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

        auto sz = (int)frames[0].elemSize();
        auto w = frames[0].cols;
        auto wsz = sz * w;
        auto maxDisparity = std::min(64, (int)(w * 0.25)) * sz;
        auto windowSize = 3 * sz;

        std::vector<uint8_t *> data(frames.size());

        for (int i = 0; i < frames.size(); ++i) {
            data[i] = frames[i].data;
        }

        auto disparityData = (int16_t *)disparity.data;
        auto roughDisparityData = (int16_t *)roughDisparity.data;
        auto rw = roughDisparity.cols;

        auto windowHeight = std::min(3, frames[0].rows);

        if (!roughDisparity.empty()) {
            CV_Assert(roughDisparity.cols == frames[0].cols / 2);
            CV_Assert(roughDisparity.rows == frames[0].rows / 2);
            CV_Assert(disparity.type() == CV_16S);
            maxDisparity = 5 * sz;
            windowSize = 3 * sz;
            windowHeight = std::min(3, frames[0].rows);
        }


        if (roughDisparity.empty()) {
            auto step = sz;
#pragma omp parallel for
            for (size_t y = 0; y < frames[0].rows - windowHeight + 1; y++) {
                windowSize = 3 * sz;
                for (int x = 0; x < maxDisparity; x += step) {
                    auto d = getDisparity(data[0], data[1], x, y, w, windowHeight, -x, maxDisparity, windowSize, sz, true);
                    disparityData[y * w + x / sz] = d;
                }
                for (int x = maxDisparity; x < wsz - windowSize - maxDisparity; x += step) {
                    auto d = getDisparity(data[0], data[1], x, y, w, windowHeight, -maxDisparity,  maxDisparity, windowSize, sz, true);
                    disparityData[y * w + x / sz] = d;
                }
                for (int x = wsz - windowSize - maxDisparity; x < wsz - windowSize; x += step) {
                    auto d = getDisparity(data[0], data[1], x, y, w, windowHeight, -maxDisparity, (wsz - windowSize - x), windowSize, sz);
                    disparityData[y * w + x / sz] = d;
                }
            }
        } else {
            auto step = sz;
#pragma omp parallel for
            for (size_t y = 0; y < frames[0].rows - windowHeight + 1; y++) {
                auto syptr = y / 2 * rw;
                for (int x = 0; x < maxDisparity; x += step) {
                    auto suggest = (int16_t)(roughDisparityData[syptr + x / sz / 2] * 2 * sz);
                    auto d = getDisparity(data[0], data[1], x, y, w, windowHeight, -x, (suggest + maxDisparity), windowSize, sz, true);
//                    disparityData[y * w + x / sz] = d;
                    disparityData[y * w + x / sz] = std::abs(d - disparityData[y * w + x / sz - 1]) == 1 ? disparityData[y * w + x / sz - 1] : d;
                }
                for (int x = maxDisparity; x < wsz - windowSize - maxDisparity; x += step) {
                    auto suggest = (int16_t)(roughDisparityData[syptr + x / sz / 2] * 2 * sz);
                    auto d = getDisparity(data[0], data[1], x, y, w, windowHeight, (suggest - maxDisparity), (suggest + maxDisparity), windowSize, sz, true);
//                    disparityData[y * w + x / sz] = d;
                    disparityData[y * w + x / sz] = std::abs(d - disparityData[y * w + x / sz - 1]) == 1 ? disparityData[y * w + x / sz - 1] : d;
                }
                for (int x = wsz - windowSize - maxDisparity; x < wsz - windowSize; x += step) {
                    auto suggest = (int16_t)(roughDisparityData[syptr + x / sz / 2] * 2 * sz);
                    auto d = getDisparity(data[0], data[1], x, y, w, windowHeight, (suggest - maxDisparity), (wsz - windowSize - x), windowSize, sz);
//                    disparityData[y * w + x / sz] = d;
                    disparityData[y * w + x / sz] = std::abs(d - disparityData[y * w + x / sz - 1]) == 1 ? disparityData[y * w + x / sz - 1] : d;
                }
            }
        }
    }

    int eval(const uint8_t *src, const uint8_t *dest, size_t windowSize, uint8_t sz);

    int16_t getDisparity(const uint8_t *data1, const uint8_t *data2, size_t x, size_t y, size_t w, size_t h, int minDisparity, int maxDisparity, size_t windowSize, uint8_t sz, bool allowFail) {
        const uint8_t *src = data1 + y * w * sz + x;
        const uint8_t *dest = data2 + y * w * sz + x;
        std::vector<int64_t> score(3, 0);
        int64_t minScore = std::numeric_limits<int64_t>::max();
        auto disparity = 0;

        auto avgScore = 0;

        int k = 0;
        for (int i = minDisparity; i < maxDisparity; i += sz) {
            int64_t newScore = 0;
            for (int j = 0; j < h; j++) {
                newScore += eval(src + j * w * sz, dest + i + j * w * sz, windowSize, sz);
            }
//            for (int j = 0; j < h; j++) {
//                newScore += eval(dest + i + j * w * sz, src + j * w * sz, windowSize, sz);
//            }

            score[k % score.size()] = newScore;

            int64_t scoreSum = 0;

            auto n = std::min(k + 1, (int) score.size());

            for (int j = 0; j < n; ++j) {
                scoreSum += score[j];
            }

            scoreSum /= n;

//            newScore += eval(dest, src - i, windowSize, sz);
            avgScore += newScore;

            if (scoreSum < minScore) {
                minScore = scoreSum;
                disparity = i / sz;
            }
            k++;
        }

        avgScore /= k;

        if (windowSize < 128 && avgScore < minScore * 2) {
            return getDisparity(data1, data2, x, y, w, h, minDisparity, maxDisparity, windowSize * 2, sz, true);
        }

        return (int16_t)(disparity);
    }

    int eval(const uint8_t *src, const uint8_t *dest, size_t windowSize, uint8_t sz) {
        auto score = 0;
        for (int i = 0; i < windowSize; ++i) {
            auto d = ((int)src[i]) - ((int)dest[i]);
            score += d * d;
        }

        return score;
    }
} // ecv