#include <imgproc.hpp>
#include <iostream>
#include "DisparityEvaluator.h"

namespace ecv {
    void DisparityEvaluator::evaluateDisparity(const std::vector<cv::Mat> &frames, cv::Mat &disparity) {
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
        auto pyramidSize = std::min((size_t)1, (size_t)std::max(0., std::floor(std::log(frames[0].cols / 400.0) / std::log(2))));
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

    void DisparityEvaluator::evaluateIncrementally(const std::vector<cv::Mat> &frames, const cv::Mat &roughDisparity, cv::Mat &disparity) {
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
        auto maxDisparity = 5 * sz;
        auto maxDisparity0 = 256 * sz;
        auto windowSize = 1 * sz;

        std::vector<uint8_t *> data(frames.size());

        for (int i = 0; i < frames.size(); ++i) {
            data[i] = frames[i].data;
        }

        auto disparityData = (int16_t *)disparity.data;
        auto rw = roughDisparity.cols;

        auto windowHeight = std::min(3, frames[0].rows);

        int16_t *roughDisparityData;

        cv::Mat roughDisparity0;

        if (!roughDisparity.empty()) {
            CV_Assert(roughDisparity.cols == frames[0].cols / 2);
            CV_Assert(roughDisparity.rows == frames[0].rows / 2);
            CV_Assert(roughDisparity.type() == CV_16S);
            roughDisparityData = (int16_t *)roughDisparity.data;
        } else {
            roughDisparity0 = cv::Mat(frames[0].cols / 2, std::max(1, frames[0].rows / 2), CV_16S);
            roughDisparity0.setTo(0);
            roughDisparityData = (int16_t *)roughDisparity0.data;
        }

        auto step = sz;
#pragma omp parallel for
        for (size_t y = 0; y < frames[0].rows - windowHeight + 1; y++) {
            auto syptr = y / 2 * rw;
            for (int x = 0; x < maxDisparity; x += step) {
                auto rd = roughDisparityData[syptr + x / sz / 2] / DISPARITY_PRECISION;
                if (rd == 0) {
                    auto d = this->getDisparity(data[1], data[0], x, y, w, windowHeight, 0, maxDisparity0, windowSize, sz);
                    disparityData[y * w + x / sz] = d;
                    continue;
                }
                auto suggest = (int16_t)(rd * 2 * sz);
                auto d = getDisparity(data[1], data[0], x, y, w, windowHeight, 0, (suggest + maxDisparity), windowSize, sz);
                disparityData[y * w + x / sz] = d;
            }
            for (int x = maxDisparity; x < wsz - windowSize - maxDisparity; x += step) {
                auto rd = roughDisparityData[syptr + x / sz / 2] / DISPARITY_PRECISION;
                if (rd == 0) {
                    auto d = getDisparity(data[1], data[0], x, y, w, windowHeight, 0,  maxDisparity0, windowSize, sz);
                    disparityData[y * w + x / sz] = d;
                    continue;
                }
                auto suggest = (int16_t)(rd * 2 * sz);
                auto d = getDisparity(data[1], data[0], x, y, w, windowHeight, (suggest - maxDisparity), (suggest + maxDisparity), windowSize, sz);
                disparityData[y * w + x / sz] = d;
            }
            for (int x = wsz - windowSize - maxDisparity; x < wsz - windowSize; x += step) {
                auto rd = roughDisparityData[syptr + x / sz / 2] / DISPARITY_PRECISION;
                if (rd == 0) {
                    auto d = getDisparity(data[1], data[0], x, y, w, windowHeight, 0, (wsz - windowSize - x), windowSize, sz);
                    disparityData[y * w + x / sz] = d;
                    continue;
                }
                auto suggest = (int16_t)(rd * 2 * sz);
                auto d = getDisparity(data[1], data[0], x, y, w, windowHeight, (suggest - maxDisparity), (wsz - windowSize - x), windowSize, sz);
                disparityData[y * w + x / sz] = d;
            }
        }
    }

    int16_t DisparityEvaluator::getDisparity(const uint8_t *data1, const uint8_t *data2, size_t x, size_t y, size_t w, size_t h, int minDisparity, int maxDisparity, size_t windowSize0, uint8_t sz) {
        const uint8_t *src = data1 + y * w * sz + x;
        const uint8_t *dest = data2 + y * w * sz + x;

        float disparity;
        float avgScore;
        float minScore;
        maxDisparity = std::max(minDisparity, maxDisparity);
        minDisparity = std::min(minDisparity, maxDisparity);
        int disparityRange = maxDisparity - minDisparity;

        auto scoreSize = std::max(2, std::min(5, disparityRange / 6));
        float score[scoreSize];
        float bestScore[scoreSize];
        memset(bestScore, 0, sizeof bestScore);
        memset(score, 0, sizeof score);

        int bestI = 0;
        int bestK = 0;

        auto wi = 0;
        int wis[] = {1, 3, 11};
        int wstep = 1;
        do {
            auto windowSize = (int)windowSize0 * wis[wi];
            minScore = std::numeric_limits<float>::max();
            disparity = 0;
            avgScore = 0;

            int k = 0;
            float scoreSum = 0;

            for (int i = minDisparity; i < maxDisparity; i += sz) {
                int64_t score0 = 0;
                for (int j = 0; j < h; j++) {
                    auto row = j * w * sz;
                    auto s0 = src + row;
                    auto d0 = dest + i + row;
#pragma omp simd
                    for (int i0 = -windowSize; i0 <= windowSize; i0+=wstep) {
                        auto e0 = s0[i0] - d0[i0];
                        score0 += e0 * e0;
                    }
                }

                auto newScore = (float)score0 / (float)windowSize;
                auto prevScore = score[k % scoreSize];
                score[k % scoreSize] = (float)newScore;

                scoreSum += (float)newScore - prevScore;

                auto n = std::min(k + 1, (int) scoreSize);

                auto scoreAvg = scoreSum / (float)n;

                avgScore += newScore;

                if (scoreAvg < minScore && k >= scoreSize) {
                    minScore = scoreAvg;
                    memcpy(bestScore, score, sizeof score);
                    bestI = i / sz;
                    bestK = k;
                }
                k++;
            }

            avgScore /= (float)k;
            wi++;
            wstep++;
        } while (wi < sizeof wis / sizeof *wis && avgScore < minScore * this->q);

        if (avgScore < minScore * this->q) {
            return 0;
        }

        auto n = std::min(bestK + 1, (int) scoreSize);
        float mass = 0;
        float sumX = 0;

        float max = 0;
        for (int i = 0; i < scoreSize; ++i) {
            max = std::max(bestScore[i], max);
        }

        for (int j = 1; j <= n; ++j) {
            auto m = max - (float)bestScore[(bestK + j) % scoreSize];
            mass += m;
            sumX += m * (float)j;
        }
        disparity = (float)bestI + (sumX / mass) - (float)n;

        return (int16_t)std::round(disparity * DisparityEvaluator::DISPARITY_PRECISION);
    }
} // ecv