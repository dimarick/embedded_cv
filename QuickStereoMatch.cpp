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
    cv::copyMakeBorder(left, leftB, borderSize, borderSize, 0, 0, cv::BORDER_REFLECT101);
    cv::copyMakeBorder(right, rightB, borderSize, borderSize, 0, 0, cv::BORDER_REFLECT101);

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

    cv::Sobel(leftF32, lGrad, CV_32F, 1, 0, 1);
    cv::Sobel(rightF32, rGrad, CV_32F, 1, 0, 1);
//
//    cv::bilateralFilter(leftF32, lGrad, 9, 75, 75);
//    cv::bilateralFilter(rightF32, rGrad, 9, 75, 75);

    imshow("lGrad", lGrad);
    imshow("rGrad", rGrad);
//    cv::cvtColor(lGrad, lGrad, COLOR_BGR2Lab, lGrad.channels());
//    cv::cvtColor(rGrad, rGrad, COLOR_BGR2Lab, lGrad.channels());

    auto gchannels = lGrad.channels();

    int width = left.cols;
    int imageStart = channels * borderSize * width;
    int gStart = lGrad.channels() * borderSize * width;

    uchar *li = leftB.getMat(AccessFlag::ACCESS_READ).data + imageStart;
    uchar *ri = rightB.getMat(AccessFlag::ACCESS_READ).data + imageStart;

    float *lg = (float *)lGrad.getMat(AccessFlag::ACCESS_READ).data + gStart;
    float *rg = (float *)rGrad.getMat(AccessFlag::ACCESS_READ).data + gStart;
    float *lm = (float *)lmap.data + borderSize * width;
    float *rm = (float *)rmap.data + borderSize * width;

#pragma omp parallel for
    for (int r = 0; r < left.rows; r++) {
        for (int c = 0; c < width; c++) {
            int o = r * width + c;
            int og = o * gchannels;
            int oi = o * channels;
            float disparity, disparity2;
            float newCost = 0;
            auto si = li + oi;
            auto sg = lg + og;
            auto ti = ri + oi;
            auto tg = rg + og;
            int shift;
            float cost = INFINITY;

            for (shift = (int)lm[o] - maxDisparity; shift <= (int)lm[o] + maxDisparity; ++shift) {
                auto s = shift * channels;

                newCost = getCost(channels, windowSize, cost, si, sg, ti + s, tg + s);

                if (newCost < cost) {
                    cost = newCost;
                    disparity = (float) shift;
                }
            }

            if (disparity - lm[o - 1] == 1) {
                lm[o] = lm[o - 1];
            } else {
                lm[o] = disparity;
            }

            cost = INFINITY;

            for (shift = (int)rm[o] - maxDisparity; shift <= (int)rm[o] + maxDisparity; ++shift) {
                auto s = shift * channels;

                newCost = getCost(channels, windowSize, cost, ti, tg, si - s, sg - s);

                if (newCost < cost) {
                    cost = newCost;
                    disparity2 = (float) shift;
                }
            }

            rm[o] = disparity2;
        }
    }
    int ld = 1, rd = 1;
    for (int r = 0; r < left.rows; r++) {
        for (int c = 0; c < width; c++) {
            int o = r * width + c;
            if ((int)lm[o - ld] - (int)lm[o] == ld) {
                lm[o] = lm[o - ld];
                ld++;
            } else {
                ld = 1;
            }
            if ((int)rm[o - rd] - (int)rm[o] == rd) {
                rm[o] = rm[o - rd];
                rd++;
            } else {
                rd = 1;
            }
//            if (lm[o] - lm[o - 2] == 2) {
//                lm[o] = lm[o - 2];
//            }
//            if (rm[o] - rm[o - 2] == -2) {
//                rm[o] = rm[o - 2];
//            }

//            auto lv = std::min(rm[o + (int) lm[o]] + lm[o], lm[o]);
//            auto rv = std::min(lm[o - (int) rm[o]] + rm[o], rm[o]);
//            lm[o] = lv;
//            rm[o] = rv;
//
//            auto ldiff = rm[o + (int) lm[o]] / lm[o];
//            auto rdiff = lm[o - (int) rm[o]] / rm[o];
//
//            ldiff = std::max(ldiff, 1 / ldiff);
//            rdiff = std::max(rdiff, 1 / rdiff);
//
//            lm[o] = ldiff > 1.4 ? std::min(rm[o + (int) lm[o]], lm[o]) : (rm[o + (int) lm[o]] / lm[o]) / 2;
//            rm[o] = rdiff > 1.4 ? std::min(lm[o - (int) rm[o]], rm[o]) : (lm[o - (int) rm[o]] / rm[o]) / 2;
        }
    }
//    for (int r = 0; r < left.rows; r++) {
//        for (int c = 0; c < width; c++) {
//            int o = r * width + c;
//
//            lm[o - (int) rm[o]] = std::max(lm[o - (int) rm[o]], rm[o]);
//            rm[o + (int) lm[o]] = std::max(rm[o + (int) lm[o]], lm[o]);
//        }
//    }

    int hist[1000];

    memset(hist, 0, sizeof(hist));
    int m = 500;

    for (int r = 0; r < left.rows; r++) {
        for (int c = 0; c < width; c++) {
            int o = r * width + c;
            hist[(int)lm[o] + m]++;
            hist[(int)rm[o] + m]++;
        }
    }

    std::cout << "";
}

float QuickStereoMatch::getCost(int channels, int windowSize, float cost, const uchar *si, const float *sg,
                          const uchar *ti, const float *tg) const {
    float chWeight[] = {1, 2, 1};
    float grWeight[] = {1, 1, 1};

    float newCost = 0;

    for (int i = 0; i <= windowSize; i++) {
        if (newCost > cost) {
            break;
        }

        auto o = i * channels;

        float iWeightX = ((float)i) / (float)windowSize;
        float iWeight = -iWeightX * iWeightX + 1;

        for (int ch = 0; ch < channels; ++ch) {
            newCost += (float)std::pow(si[o + ch] - ti[o + ch], 2) * chWeight[ch] * iWeight;
            newCost += (float)std::pow(sg[o + ch] - tg[o + ch], 2) * grWeight[ch] * iWeight * 10;
//            newCost -= (float)std::pow(ti[ch] - ti[o + ch], 2) * 0.1;
//            newCost -= (float)std::abs(tg[ch] - tg[o + ch]) * grWeight[ch] * 0.5;
//            sum += (float)std::abs(tg[ch] - tg[i + ch]);
        }
    }

    return newCost;
}
