//
// Created by dima on 30.06.24.
//

#include "QuickStereoMatch.h"
#include <opencv2/core/mat.hpp>
#include <iostream>

void QuickStereoMatch::computeDisparityMap(const Mat &left, const Mat &right, Mat &lmap, Mat &rmap, int maxDisparity, int windowSize, int borderSize) {
    assert(left.size == right.size);
    assert(left.channels() == right.channels());
    assert(left.depth() == CV_8U);
    assert(right.depth() == CV_8U);

    assert(lmap.type() == CV_32FC1 || lmap.empty());
    assert(rmap.type() == CV_32FC1 || rmap.empty());

    UMat lGrad, rGrad;

    UMat leftB, rightB, leftF32, rightF32;

    int channels = left.channels();

    // Добавляем строку сверху и снизу чтобы безопасно адресовать слишком большие и отрицательные смещения
    cv::copyMakeBorder(left, leftB, borderSize, borderSize, maxDisparity, maxDisparity, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(right, rightB, borderSize, borderSize, maxDisparity, maxDisparity, cv::BORDER_REPLICATE);

//    map = Mat::ones(leftB.size(), CV_32FC1);
//    lmap = Mat::zeros(leftB.size(), CV_32FC1);
//    rmap = Mat::zeros(rightB.size(), CV_32FC1);
    if (lmap.empty()) {
        lmap = Mat::zeros(leftB.size(), CV_32FC1);
    }

    if (rmap.empty()) {
        rmap = Mat::zeros(rightB.size(), CV_32FC1);
    }

    leftB.convertTo(leftF32, CV_32F);
    rightB.convertTo(rightF32, CV_32F);

    cv::Sobel(leftF32, lGrad, CV_32F, 1, 0, 3);
    cv::Sobel(rightF32, rGrad, CV_32F, 1, 0, 3);
//
//    cv::cvtColor(lGrad, lGrad, COLOR_BGR2Lab, lGrad.channels());
//    cv::cvtColor(rGrad, rGrad, COLOR_BGR2Lab, lGrad.channels());

    auto gchannels = lGrad.channels();

    int width = leftB.cols;
    int imageStart = channels * borderSize * width;
    int gStart = lGrad.channels() * borderSize * width;

    uchar *li = leftB.getMat(AccessFlag::ACCESS_READ).data + imageStart;
    uchar *ri = rightB.getMat(AccessFlag::ACCESS_READ).data + imageStart;

    float *lg = (float *)lGrad.getMat(AccessFlag::ACCESS_READ).data + gStart;
    float *rg = (float *)rGrad.getMat(AccessFlag::ACCESS_READ).data + gStart;
    float *lm = (float *)lmap.data + borderSize * width;
    float *rm = (float *)rmap.data + borderSize * width;

    std::vector<float> costs;
    std::vector<bool> lxrefs;
    std::vector<bool> rxrefs;

    costs.resize(maxDisparity * 2);
    lxrefs.resize(width + maxDisparity * 2);
    rxrefs.resize(width + maxDisparity * 2);

#pragma omp parallel for
    for (int r = 0; r < left.rows; r++) {
        std::fill(lxrefs.begin(), lxrefs.end(), false);
        std::fill(rxrefs.begin(), rxrefs.end(), false);

        for (int c = 0; c < width; c++) {
            int o = r * width + c;
            int og = o * gchannels;
            int oi = o * channels;
            float disparity = -maxDisparity, disparity2 = -maxDisparity;
            float newCost = 0;
            auto si = li + oi;
            auto sg = lg + og;
            auto ti = ri + oi;
            auto tg = rg + og;
            int shift;
            float cost = INFINITY;

            for (shift = (int)lm[o] - maxDisparity; shift <= (int)lm[o] + maxDisparity; ++shift) {
                auto s = shift * channels;

                costs[shift + maxDisparity] = newCost = getCost(channels, windowSize, cost, si, sg, ti + s, tg + s, width * channels);

                if (newCost < cost) {
                    cost = newCost;
                    disparity = (float) shift;
                }
            }

            lm[o] = disparity;

            cost = INFINITY;

            for (shift = (int)rm[o] - maxDisparity; shift <= (int)rm[o] + maxDisparity; ++shift) {
                auto s = shift * channels;

                costs[shift + maxDisparity] = newCost = getCost(channels, windowSize, cost, ti, tg, si - s, sg - s, width * channels);

                if (newCost < cost) {
                    cost = newCost;
                    disparity2 = (float) shift;
                }
            }

            rm[o] = disparity2;
        }
    }

#pragma omp parallel for
    for (int r = 0; r < left.rows; r++) {
        int ld = 1, rd = 1;
        for (int c = 0; c < width; c++) {
            int o = r * width + c;
            if ((int)lm[o - ld] - (int)lm[o] == ld) {
                lm[o] = lm[o - ld];
                ld++;
            } else {
                ld = 1;
            }
        }
        for (int c = 0; c < width; c++) {
            int o = r * width + width - c - 1;

            if ((int)rm[o + rd] - (int)rm[o] == rd) {
                rm[o] = rm[o + rd];
                rd++;
            } else {
                rd = 1;
            }
        }
        for (int c = 0; c < width; c++) {
            int o = r * width + c;

            auto lv = std::max(rm[o + (int) lm[o]] + lm[o], lm[o]);
            auto rv = std::max(lm[o - (int) rm[o]] + rm[o], rm[o]);
            lm[o] = (lm[o] + lv) / 2;
            rm[o] = (rm[o] + rv) / 2;
        }
    }
}

float QuickStereoMatch::getCost(int channels, int windowSize, float cost, const uchar *si, const float *sg,
                          const uchar *ti, const float *tg, const int rowWidth) const {
    float chWeight[] = {1, 2, 1};
    float grWeight[] = {10, 10, 10};

    float newCost = 0;

    float llight = 0, rlight = 0, grQ = 0;

    for (int i = 0; i <= windowSize; i++) {
        auto o = i * channels;
        for (int ch = 0; ch < channels; ++ch) {
            llight += (float)si[o + ch];
            rlight += (float)ti[o + ch];
            grQ -= std::abs(sg[o + 1] - sg[o + ch]);
            grQ += std::abs(sg[o + ch]) * 0.7f;
        }
    }

    rlight /= llight;
    llight = 1;

    for (int i = 0; i <= windowSize; i++) {
        if (newCost > cost) {
            break;
        }

        auto o = i * channels;

        for (int ch = 0; ch < channels; ++ch) {
            newCost += (float)std::pow((float)si[o + ch] - (float)ti[o + ch] / rlight, 2) * chWeight[ch] * grQ;
            newCost += (float)std::pow((float)sg[o + ch] - (float)tg[o + ch], 2) * grWeight[ch] * grQ;
//            newCost += (float)std::pow((float)si[o + ch + rowWidth] - (float)ti[o + ch + rowWidth], 2) * chWeight[ch] * grQ * 0.5;
//            newCost += (float)std::pow((float)sg[o + ch + rowWidth] - (float)tg[o + ch + rowWidth] / rlight, 2) * grWeight[ch] * grQ * 0.5;
//            newCost += (float)std::pow((float)si[o + ch - rowWidth] - (float)ti[o + ch - rowWidth], 2) * chWeight[ch] * grQ * 0.5;
//            newCost += (float)std::pow((float)sg[o + ch - rowWidth] - (float)tg[o + ch - rowWidth] / rlight, 2) * grWeight[ch] * grQ * 0.5;
        }
    }

    return newCost;
}
