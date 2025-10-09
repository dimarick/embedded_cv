//
// Created by dima on 30.06.24.
//

#include "QuickStereoMatch.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

void QuickStereoMatch::computeDisparityMap(const Mat &left, const Mat &right, Mat &lmap, Mat &rmap, int maxDisparity, int windowSize, int borderSize) {
    assert(left.size == right.size);
    assert(left.channels() == right.channels());
    assert(left.depth() == CV_8U);
    assert(right.depth() == CV_8U);

    assert(lmap.type() == CV_32FC1 || lmap.empty());
    assert(rmap.type() == CV_32FC1 || rmap.empty());

    UMat lGrad, rGrad, lGrad2, rGrad2, lGradMaskS, rGradMaskS, lGradMaskV, rGradMaskV;

    UMat leftB, rightB, leftF32, rightF32;

    int channels = left.channels();

    // Добавляем строку сверху и снизу чтобы безопасно адресовать слишком большие и отрицательные смещения
    cv::copyMakeBorder(left, leftB, borderSize, borderSize, maxDisparity, maxDisparity, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(right, rightB, borderSize, borderSize, maxDisparity, maxDisparity, cv::BORDER_REPLICATE);

//    map = Mat::ones(leftB.size(), CV_32FC1);
//    lmap = Mat::zeros(leftB.size(), CV_32FC1);
//    rmap = Mat::zeros(rightB.size(), CV_32FC1);
    if (lmap.empty()) {
        lmap = Mat(leftB.size(), CV_32FC1, Scalar::all(-maxDisparity));
    }

    if (rmap.empty()) {
        rmap = Mat(leftB.size(), CV_32FC1, Scalar::all(-maxDisparity));
    }

    leftB.convertTo(leftF32, CV_32F);
    rightB.convertTo(rightF32, CV_32F);

    cv::Sobel(leftF32, lGrad, CV_32F, 1, 0, 3);
    cv::Sobel(rightF32, rGrad, CV_32F, 1, 0, 3);
    cv::Sobel(leftF32, lGrad2, CV_32F, 0, 1, 3);
    cv::Sobel(rightF32, rGrad2, CV_32F, 0, 1, 3);
    cv::multiply(lGrad, lGrad, lGrad, 1. / 256);
    cv::multiply(rGrad, rGrad, rGrad, 1. / 256);
    cv::multiply(lGrad2, lGrad2, lGrad2, 1. / 256);
    cv::multiply(rGrad2, rGrad2, rGrad2, 1. / 256);
    cv::multiply(lGrad, lGrad2, lGrad);
    cv::multiply(rGrad, rGrad2, rGrad);

    Mat hist;
    double lgMax;
    double rgMax;
    cv::minMaxLoc(lGrad, nullptr, &lgMax, nullptr, nullptr, noArray());
    cv::minMaxLoc(rGrad, nullptr, &rgMax, nullptr, nullptr, noArray());

    cv::threshold(lGrad, lGrad, 4.5, lgMax, cv::THRESH_TOZERO);
    cv::threshold(rGrad, rGrad, 4.5, rgMax, cv::THRESH_TOZERO);


//    this->ridgeFilter(leftF32, lGrad);
//    this->ridgeFilter(rightF32, rGrad);

//    cv::threshold(lGrad, lGrad, 128, 512, THRESH_BINARY);
//    cv::threshold(rGrad, rGrad, 128, 512, THRESH_BINARY);

//
//    cv::Sobel(leftF32, lGrad, CV_32F, 1, 0, 3);
//    cv::Sobel(rightF32, rGrad, CV_32F, 1, 0, 3);
//
//    imshow("lGrad", lGradMaskV);
//    imshow("rGrad", rGradMaskV);

//
//    cv::cvtColor(lGrad, lGrad, COLOR_BGR2Lab, lGrad.channels());
//    cv::cvtColor(rGrad, rGrad, COLOR_BGR2Lab, lGrad.channels());

    auto gchannels = lGrad.channels();

    int width = leftB.cols;
    int imageStart = channels * (borderSize * width + maxDisparity);
    int gStart = lGrad.channels() * (borderSize * width + maxDisparity);

    uchar *li = leftB.getMat(AccessFlag::ACCESS_READ).data + imageStart;
    uchar *ri = rightB.getMat(AccessFlag::ACCESS_READ).data + imageStart;

    float *lg = (float *)lGrad.getMat(AccessFlag::ACCESS_READ).data + gStart;
    float *rg = (float *)rGrad.getMat(AccessFlag::ACCESS_READ).data + gStart;
    float *lm = (float *)lmap.data + borderSize * width + maxDisparity;
    float *rm = (float *)rmap.data + borderSize * width + maxDisparity;
//
//    std::vector<cv::UMat> grad1 = {lGrad};
//    std::vector<cv::UMat> grad2 = {rGrad};
//    std::vector<int> channelsHist = {0};
//    std::vector<int> histSize = {32};
//    std::vector<float> ranges = {-512, 512};
//    cv::Mat histL, histR;
//
//    cv::calcHist(grad1, channelsHist, noArray(), histL, histSize, ranges);
//    cv::calcHist(grad2, channelsHist, noArray(), histR, histSize, ranges);

    std::vector<unsigned char> lSegmentsMask(left.rows * width), rSegmentsMask(left.rows * width);

    Mat lSegmentsMaskMat(left.rows, width, CV_8UC1, lSegmentsMask.data());
    Mat rSegmentsMaskMat(left.rows, width, CV_8UC1, rSegmentsMask.data());

#pragma omp parallel for
    for (int r = 0; r < left.rows; r++) {
        int currentPeak = 0;
        for (int c = 0; c < width; c++) {
            bool lPeak = false, rPeak = false;
            int o = r * width + c;
            for (int ch = 0; ch < gchannels; ++ch) {
                int og = o * gchannels + ch;

                bool peakV = true, peakH = true;
                for (int v = 1; v <= 2 && peakV; ++v) {
                    auto z = std::max(0, v - 1);
                    for (int dv = -1; dv <= 1 && peakV; dv += 2) {
                        peakV = peakV
                            && lg[og + (dv * z * width) * gchannels]
                             > lg[og + (dv * v * width) * gchannels];
                    }
                }

                for (int h = 1; h <= 2 && peakH; ++h) {
                    auto zh = std::max(0, h - 1);
                    for (int dh = -1; dh <= 1 && peakH; dh += 2) {
                        peakH = peakH
                            && lg[og + (dh * zh) * gchannels]
                             > lg[og + (dh * h) * gchannels];
                    }
                }

                lPeak = lPeak || peakV || peakH;

//                sign = (rg[og] + 0.1f) / std::abs(rg[og] + 0.1f);

                peakV = true, peakH = true;
                for (int v = 1; v <= 2 && peakV; ++v) {
                    auto z = v - 1;
                    for (int dv = -1; dv <= 1; dv += 2) {
                        peakV = peakV
                            && rg[og + (dv * z * width) * gchannels]
                             > rg[og + (dv * v * width) * gchannels];
                    }
                }

                for (int h = 1; h <= 2; ++h) {
                    auto zh = h - 1;
                    for (int dh = -1; dh <= 1 && peakH; dh += 2) {
                        peakH = peakH
                            && rg[og + (dh * zh) * gchannels]
                             > rg[og + (dh * h) * gchannels];
                    }
                }

                rPeak = rPeak || peakV || peakH;
            }

            lSegmentsMask[o] = lPeak;
            rSegmentsMask[o] = rPeak;

            currentPeak += rPeak || lPeak;
        }
    }

//    lSegmentsMaskMat *= 255;
//    rSegmentsMaskMat *= 255;

    Mat kernel = Mat::ones( 5, 5, CV_32F );

    UMat lSegmentsMaskMat2, rSegmentsMaskMat2;
    lSegmentsMaskMat.copyTo(lSegmentsMaskMat2);
    rSegmentsMaskMat.copyTo(rSegmentsMaskMat2);

    cv::filter2D(lSegmentsMaskMat2, lSegmentsMaskMat2, kernel);
    cv::filter2D(rSegmentsMaskMat2, rSegmentsMaskMat2, kernel);
    cv::threshold(lSegmentsMaskMat2, lSegmentsMaskMat2, 3.5, 25, cv::THRESH_BINARY);
    cv::threshold(rSegmentsMaskMat2, rSegmentsMaskMat2, 3.5, 25, cv::THRESH_BINARY);
    cv::bitwise_and(lSegmentsMaskMat, lSegmentsMaskMat2, lSegmentsMaskMat2);
    cv::bitwise_and(rSegmentsMaskMat, rSegmentsMaskMat2, rSegmentsMaskMat2);

//    Mat mk = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(2, 2));
//    cv::morphologyEx(lSegmentsMaskMat2, lSegmentsMaskMat2, cv::MORPH_CLOSE, mk, Point(-1, -1), 4);
//    cv::morphologyEx(rSegmentsMaskMat2, rSegmentsMaskMat2, cv::MORPH_CLOSE, mk, Point(-1, -1), 4);

    lSegmentsMaskMat2.copyTo(lSegmentsMaskMat);
    rSegmentsMaskMat2.copyTo(rSegmentsMaskMat);

    imshow("Left Mask", lSegmentsMaskMat * 255);
    imshow("Right Mask", rSegmentsMaskMat * 255);
    imshow("rGrad", rGrad);

#pragma omp parallel for
    for (int r = 0; r < left.rows; r++) {
        auto w = width * channels;
//        std::vector<float> costs((maxDisparity * 2 + 1) * 2);
//        std::vector<float> costs2((maxDisparity * 2 + 1) * 5 * 2);
//        cv::Mat mCosts(costs, false), mCosts2(costs2, false);
//        mCosts = mCosts.reshape(1, 2);
//        mCosts2 = mCosts2.reshape(1, 2);
//        float * costsL = costs.data();
//        float * costsR = costs.data() + (int)costs.size() / 2;

//        std::vector<int> lSegments(width), rSegments(width);
//
//        int currentPeak = 0;
//        for (int c = 0; c < width; c++) {
//            bool lPeak = false, rPeak = false;
//            for (int ch = 0; ch < gchannels; ++ch) {
//                int o = (r * width + c) * gchannels + ch;
//                float sign;
//                bool isNotNoise;
//
//                sign = (lg[o] + 0.1f) / std::abs(lg[o] + 0.1f);
//
//                isNotNoise =
//                           sign * lg[o] > sign * (lg[o - 1 * gchannels])
//                        && sign * lg[o] > sign * (lg[o + 1 * gchannels])
//                        && sign * (lg[o - 1 * gchannels]) > sign * (lg[o - 2 * gchannels])
//                        && sign * (lg[o + 1 * gchannels]) > sign * (lg[o + 2 * gchannels])
//                        && sign * (lg[o - 2 * gchannels]) > sign * (lg[o - 3 * gchannels])
//                        && sign * (lg[o + 2 * gchannels]) > sign * (lg[o + 3 * gchannels]);
//
//                lPeak = lPeak || isNotNoise;
//
//                sign = (rg[o] + 0.1f) / std::abs(rg[o] + 0.1f);
//
//                isNotNoise =
//                        sign * rg[o] > sign * (rg[o - 1 * gchannels])
//                        && sign * rg[o] > sign * (rg[o + 1 * gchannels])
//                        && sign * (rg[o - 1 * gchannels]) > sign * (rg[o - 2 * gchannels])
//                        && sign * (rg[o + 1 * gchannels]) > sign * (rg[o + 2 * gchannels])
//                        && sign * (rg[o - 2 * gchannels]) > sign * (rg[o - 3 * gchannels])
//                        && sign * (rg[o + 2 * gchannels]) > sign * (rg[o + 3 * gchannels]);
//
//                rPeak = rPeak || isNotNoise;
//            }
//
//            lSegments[currentPeak] = c * lPeak;
//            rSegments[currentPeak] = c * rPeak;
//
//            currentPeak += (rPeak || lPeak) && currentPeak < width;
//
//            lm[r * width + c] = -maxDisparity;
//            rm[r * width + c] = -maxDisparity;
//        }

        unsigned char*lSegments = &lSegmentsMask[r * width];
        unsigned char*rSegments = &rSegmentsMask[r * width];

        for (int c = 0; c < width; c++) {
            auto isLeft = lSegments[c] > 0;
            auto isRight = rSegments[c] > 0;

            if (!isLeft && !isRight) {
                continue;
            }

            int o = r * width + c;
            int og = o * gchannels;
            int oi = o * channels;
            float disparityL1 = -maxDisparity, disparityR1 = -maxDisparity, disparityL2 = -maxDisparity, disparityR2 = -maxDisparity;
            float newCostL1 = 0, newCostR1 = 0, newCostL2 = 0, newCostR2 = 0;
            auto si = li + oi;
            auto sg = lg + og;
            auto ti = ri + oi;
            auto tg = rg + og;
            int shift;
            float costL1 = INFINITY, costR1 = INFINITY;
//
//            if (isLeft) {
//                lm[o] = maxDisparity / 2;
//            }
//            if (isRight) {
//                rm[o] = maxDisparity / 2;
//            }

            for (shift = (int)-maxDisparity; shift <= (int)maxDisparity; ++shift) {
                auto s = shift * channels;

                if (isLeft) {
                    newCostL1 = getCost(channels, windowSize, costL1, si, sg, ti + s, tg + s, width * channels);
//                newCostL2 = getCost(channels, windowSize, costL2, si - w, sg - w, ti + s - w, tg + s - w, width * channels);

                    if (newCostL1 < costL1) {
                        costL1 = newCostL1;
                        disparityL1 = (float) shift;
                    }

//                if (newCostL2 < costL2) {
//                    costL2 = newCostL2;
//                    disparityL2 = (float) shift;
//                }
                }
                if (isRight) {
                    newCostR1 = getCost(channels, windowSize, costR1, ti, tg, si - s, sg - s, width * channels);
//                newCostR1 = getCost(channels, windowSize, costR1, ti - w, tg - w, si - s - w, sg - s - w, width * channels);

                    if (newCostR1 < costR1) {
                        costR1 = newCostR1;
                        disparityR1 = (float) shift;
                    }
//
//                if (newCostR2 < costR2) {
//                    costR2 = newCostR2;
//                    disparityR2 = (float) shift;
//                }
//               subpixel precision
//                costsL[shift + maxDisparity] = newCostL1;
//                costsR[shift + maxDisparity] = newCostR1;
                }
            }

//            cv::resize(mCosts, mCosts2, Size(0, 0), 5, 1, INTER_LINEAR);

            lm[o] = disparityL1;
            rm[o] = disparityR1;
        }

//        for (int c = 0; c < width; c++) {
//            int o = r * width + c;
//            int og = o * gchannels;
//            int oi = o * channels;
//            float disparityL1 = -maxDisparity, disparityR1 = -maxDisparity, disparityL2 = -maxDisparity, disparityR2 = -maxDisparity;
//            float newCostL1 = 0, newCostR1 = 0, newCostL2 = 0, newCostR2 = 0;
//            auto si = li + oi;
//            auto sg = lg + og;
//            auto ti = ri + oi;
//            auto tg = rg + og;
//            int shift;
//            float costL1 = INFINITY, costR1 = INFINITY;
//
//            for (shift = (int)-maxDisparity; shift <= (int)maxDisparity; ++shift) {
//                auto s = shift * channels;
//
//                newCostL1 = getCost(channels, windowSize, costL1, si, sg, ti + s, tg + s, width * channels);
////                newCostL2 = getCost(channels, windowSize, costL2, si - w, sg - w, ti + s - w, tg + s - w, width * channels);
//
//                if (newCostL1 < costL1) {
//                    costL1 = newCostL1;
//                    disparityL1 = (float) shift;
//                }
//
////                if (newCostL2 < costL2) {
////                    costL2 = newCostL2;
////                    disparityL2 = (float) shift;
////                }
//
//                newCostR1 = getCost(channels, windowSize, costR1, ti, tg, si - s, sg - s, width * channels);
////                newCostR2 = getCost(channels, windowSize, costR2, ti - w, tg - w, si - s - w, sg - s - w, width * channels);
//
//                if (newCostR1 < costR1) {
//                    costR1 = newCostR1;
//                    disparityR1 = (float) shift;
//                }
////
////                if (newCostR2 < costR2) {
////                    costR2 = newCostR2;
////                    disparityR2 = (float) shift;
////                }
////               subpixel precision
////                costsL[shift + maxDisparity] = newCostL1;
////                costsR[shift + maxDisparity] = newCostR1;
//            }
//
////            cv::resize(mCosts, mCosts2, Size(0, 0), 5, 1, INTER_LINEAR);
//
//            lm[o] = disparityL1;
//            rm[o] = disparityR1;
//
////            if (std::abs(disparityL1 - disparityL2) <= 0) {
////                lm[o] = (disparityL1 + disparityL2) / 2;
////            } else {
////                lm[o] = -maxDisparity - 1;
////            }
////
////            if (std::abs(disparityR1 - disparityR2) <= 0) {
////                rm[o] = (disparityR1 + disparityR2) / 2;
////            } else {
////                rm[o] = -maxDisparity - 1;
////            }
//        }
    }

//#pragma omp parallel for
    for (int r = 0; r < left.rows; r++) {
//
//        auto lastValid = r * width;
//        std::vector<float> iBase(channels);
//        std::vector<float> iStep(channels);
//        for (int c = 0; c < width; c++) {
//            int o = r * width + c;
//            if (lm[o] <= -maxDisparity && lm[o - 1] > -maxDisparity) {
//                lastValid = o - 1;
//            } else if (lm[o - 1] <= -maxDisparity && lm[o] > -maxDisparity) {
//                auto missingWidth = o - lastValid;
//                if (missingWidth > 0.01 * width) {
//                    continue;
//                }
//                auto base = lm[lastValid];
//                for (int i = 0; i < channels; ++i) {
//                    iBase[i] = li[(lastValid) * channels + i];
//                    iStep[i] = lm[o] / (li[(o) * channels + i] + base - iBase[i]) / missingWidth;
//                }
//
//                for (int c1 = 0; c1 < missingWidth; ++c1) {
//                    auto pixelSum = 0.f;
//                    for (int i = 0; i < channels; ++i) {
//                        uchar pixel = li[(lastValid + c1) * channels + i];
//                        pixelSum += (pixel - iBase[i]) * (iStep[i] * c1) + base;
//                    }
//                    lm[lastValid + c1] = pixelSum / channels;
//                }
//            }
//        }

//      performing cross-check
        for (int c = 0; c < width; c++) {
            int o = r * width + c;

            auto lxv = rm[o + (int) lm[o] + maxDisparity];
            if (lxv > -maxDisparity) {
                lm[o] = std::min(lm[o], lxv);
            }

            auto rxv = lm[o - (int) rm[o] - maxDisparity];
            if (rxv > -maxDisparity) {
                rm[o] = std::min(rm[o], rxv);
            }
        }

//        int ld = 1, rd = 1;
//        for (int c = 0; c < width; c++) {
//            int o = r * width + width - c - 1;
//            if ((int)lm[o + ld] - (int)lm[o] == -ld) {
//                ld++;
//                lm[o] = lm[o + ld];
//            } else if (ld > 1) {
//                lm[o + ld] = lm[o + ld + 1];
//                ld = 1;
//            }
//        }

//        for (int c = 0; c < width; c++) {
//            int o = r * width + width - c - 1;
//
//            if ((int)rm[o + rd] - (int)rm[o] == rd) {
//                rm[o] = rm[o + rd];
//                rd++;
//            } else {
//                rd = 1;
//            }
//        }
    }
}

float QuickStereoMatch::getCost(int channels, int windowSize, float cost, const uchar *si, const float *sg,
                          const uchar *ti, const float *tg, const int rowWidth) const {
    float chWeight[] = {1, 2, 1};
    float grWeight[] = {10, 20, 10};

    float newCost = 0;

    float llight = 0, rlight = 0, grQ = 1;

    for (int i = 0; i <= windowSize; i++) {
        auto o = i * channels;
        for (int ch = 0; ch < channels; ++ch) {
            llight += (float)si[o + ch];
            rlight += (float)ti[o + ch];
//            grQ -= std::abs(sg[o + 1] - sg[o + ch]);
//            grQ += std::abs(sg[o + ch]) * 0.5;
        }
    }

//    grQ = std::max(grQ, 0.f);
//
//    if (grQ == 0) {
//        return INFINITY;
//    }

    auto eps = 0.5f;

    rlight = (rlight + eps) / (llight + eps);
    llight = 1;

    for (int i = 0; i <= windowSize; i++) {
        if (newCost > cost) {
            break;
        }

        auto o = i * channels;

        for (int ch = 0; ch < channels; ++ch) {
            newCost += (float)std::pow((float)si[o + ch] - (float)ti[o + ch] / rlight, 2) * chWeight[ch] * grQ;
//            newCost += (float)std::pow((float)sg[o + ch] - (float)tg[o + ch], 2) * grWeight[ch] * grQ;
//            newCost += (float)std::pow((float)si[o + ch + rowWidth] - (float)ti[o + ch + rowWidth], 2) * chWeight[ch] * grQ * 0.5;
//            newCost += (float)std::pow((float)sg[o + ch + rowWidth] - (float)tg[o + ch + rowWidth] / rlight, 2) * grWeight[ch] * grQ * 0.5;
//            newCost += (float)std::pow((float)si[o + ch - rowWidth] - (float)ti[o + ch - rowWidth], 2) * chWeight[ch] * grQ * 0.5;
//            newCost += (float)std::pow((float)sg[o + ch - rowWidth] - (float)tg[o + ch - rowWidth] / rlight, 2) * grWeight[ch] * grQ * 0.5;
        }
    }

    return newCost;
}

void QuickStereoMatch::ridgeFilter(UMat img, UMat &out) {
    CV_Assert(img.channels() == 1 || img.channels() == 3);

    UMat sbx, sby;
    Sobel(img, sbx, CV_32F, 1, 0, 3, 1, 0);
    Sobel(img, sby, CV_32F, 0, 1, 3, 1, 0);

    UMat sbxx, sbyy, sbxy;
    Sobel(sbx, sbxx, CV_32F, 1, 0, 3, 1, 0);
    Sobel(sby, sbyy, CV_32F, 0, 1, 3, 1, 0);
    Sobel(sbx, sbxy, CV_32F, 0, 1, 3, 1, 0);

    UMat sb2xx, sb2yy, sb2xy;
    multiply(sbxx, sbxx, sb2xx);
    multiply(sbyy, sbyy, sb2yy);
    multiply(sbxy, sbxy, sb2xy);

    UMat sbxxyy;
    multiply(sbxx, sbyy, sbxxyy);

    UMat rootex;
//    rootex = (sb2xx +  (sb2xy + sb2xy + sb2xy + sb2xy)  - (sbxxyy + sbxxyy) + sb2yy );
//    rootex = sb2xx + sb2xy + sb2xy + sb2xy + sb2xy - sbxxyy - sbxxyy + sb2yy;

    cv::add(sb2xx, sb2xy, rootex);
    cv::add(rootex, sb2xy, rootex);
    cv::add(rootex, sb2xy, rootex);
    cv::add(rootex, sb2xy, rootex);
    cv::add(rootex, sb2yy, rootex);
    cv::subtract(rootex, sbxxyy, rootex);
    cv::subtract(rootex, sbxxyy, rootex);

    UMat root;
    sqrt(rootex, root);
//    out = ( sbxx + sbyy + root );

    cv::add(sbxx, sbyy, out);
    cv::add(out, root, out);
}
