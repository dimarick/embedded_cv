#include "opencl_ide_defs.h"

float getDisparity(
        uchar *data1,
        uchar *data2,
        size_t x,
        size_t y,
        size_t w,
        size_t h,
        int minDisparity,
        int maxDisparity,
        size_t windowSize0,
        int sz,
        float* q
) {
    const uchar *src = data1 + y * w * sz + x;
    const uchar *dest = data2 + y * w * sz + x;

    float disparity;
    float avgScore;
    float maxScore;
    maxDisparity = max(minDisparity, maxDisparity);
    minDisparity = min(minDisparity, maxDisparity);
    int disparityRange = maxDisparity - minDisparity;

    const int maxScoreSize = 5;
    int scoreSize = max(2, min(maxScoreSize, disparityRange / 6));
    float score[maxScoreSize];
    float bestScore[maxScoreSize];

    for (int i = 0; i < scoreSize; ++i) {
        bestScore[i] = 0;
        score[i] = 0;
    }

    int bestI = 0;
    int bestK = 0;

    int wi = 0;
    int wis[] = {1};
    int wstep = 1;
    int k = 0;
//    do {
        int windowSize = (int)windowSize0 * wis[wi];
        float maxPossibleScore = 255.f * 255.f * (float)windowSize * (float)h;
        maxScore = 0;
        avgScore = 0;

        k = 0;
        float scoreSum = 0;

        for (int i = minDisparity; i < maxDisparity; i += sz) {
            float score0 = 0;

            int hstep = w * sz;
            for (int j = 0; j < h * hstep; j+= hstep) {
                for (int i0 = 0; i0 <= windowSize; i0+=wstep) {
                    const float d = src[j + i0] - dest[i + j + i0];
                    score0 += d * d;
                }
            }

            float newScore = (maxPossibleScore - score0) / (float)windowSize;
            float prevScore = score[k % scoreSize];
            score[k % scoreSize] = newScore;

            scoreSum += newScore - prevScore;

            int n = min(k + 1, (int) scoreSize);

            float currentScore = scoreSum / (float)n;

            avgScore += newScore;

            bool needUpdate = currentScore > maxScore;

            bestI = needUpdate ? i / sz : bestI;
            bestK = needUpdate ? k : bestK;
            maxScore = fmax(maxScore, currentScore);
            for (int j = 0; j < scoreSize; ++j) {
                bestScore[j] = needUpdate ? score[j] : bestScore[j];
            }

            k++;
        }
//
//        avgScore /= (float)k;
//        wi++;
//    } while (wi < sizeof wis / sizeof *wis && avgScore > maxScore * 0.9);

//    if (avgScore > maxScore * *q) {
//        *q = *q + 3e-7;
//        return 0;
//    }
//
//    *q = *q - 1e-7;

    scoreSize = min(k, scoreSize);
    int n = min(bestK + 1, scoreSize);
    float mass = 0;
    float sumX = 0;

    float jf = 1;
    for (int j = 1; j <= n; ++j, ++jf) {
        float m = bestScore[(bestK + j) % scoreSize];
        mass += m;
        sumX += m * jf;
    }
    disparity = (float)(bestI - n) + (sumX / mass);

    return disparity;
}

float16 getDisparityV16(
        uchar *data1,
        uchar *data2,
        size_t x,
        size_t y,
        size_t w,
        size_t h,
        int minDisparity,
        int maxDisparity,
        size_t windowSize0,
        int sz,
        float* q,
        bool debug
) {
    const uchar *src = data1 + y * w * sz + x;
    const uchar *dest = data2 + y * w * sz + x;

    float16 disparity;
    float16 avgScore;
    float16 maxScore;
    maxDisparity = max(minDisparity, maxDisparity);
    minDisparity = min(minDisparity, maxDisparity);
    int disparityRange = maxDisparity - minDisparity;

    const int maxScoreSize = 5;
    int scoreSize = max(2, min(maxScoreSize, disparityRange / 6));
    float16 score[maxScoreSize];
    float16 bestScore[maxScoreSize];

    for (int i = 0; i < scoreSize; ++i) {
        bestScore[i] = 0;
        score[i] = 0;
    }

    int16 bestI = 0;
    int16 bestK = 0;

    int wi = 0;
    int wis[] = {1};
    int wstep = 1;
    int k = 0;
//    do {
        int windowSize = (int)windowSize0 * wis[wi];
        float maxPossibleScore = 255.f * 255.f * (float)windowSize * (float)h;
        maxScore = (float16)0;
        avgScore = (float16)0;

        k = 0;
        float16 scoreSum = 0;

        for (int i = minDisparity; i < maxDisparity; i += sz) {
            float16 score0 = (float16)0;

            int hstep = w * sz;
            for (int j = 0; j < h * hstep; j+= hstep) {
                for (int i0 = 0; i0 <= windowSize / 16 + 1; i0+=wstep) {
                    const float16 d0 = convert_float16(vload16(i0, src + j));
                    const float16 d1 = convert_float16(vload16(i0, dest + i + j));
                    const float16 d = d0 - d1;
                    score0 += d * d;
                }
            }

            float16 newScore = (maxPossibleScore - score0) / (float)windowSize;
            float16 prevScore = score[k % scoreSize];
            score[k % scoreSize] = newScore;

            scoreSum += newScore - prevScore;

            int n = min(k + 1, (int) scoreSize);

            float16 currentScore = scoreSum / (float)n;

            avgScore += newScore;

            int16 needUpdate = isgreater(currentScore, maxScore);

            bestI = select(bestI, (int16)i / sz, needUpdate);
            bestK = select(bestK, (int16)k, needUpdate);
            maxScore = fmax(maxScore, currentScore);

            for (int j = 0; j < scoreSize; ++j) {
                bestScore[j] = select(bestScore[j], score[j], needUpdate);
            }

            k++;
        }
//
//        avgScore /= (float)k;
//        wi++;
//    } while (wi < sizeof wis / sizeof *wis && avgScore > maxScore * 0.9);

//    if (avgScore > maxScore * *q) {
//        *q = *q + 3e-7;
//        return 0;
//    }
//
//    *q = *q - 1e-7;

    scoreSize = min(k, scoreSize);
    float16 mass = 0;
    float16 sumX = 0;

    float jf = 1;
    for (int j = 1; j <= scoreSize; ++j, ++jf) {
        float16 m = (float16)(
            bestScore[(bestK.s0 + j) % scoreSize].s0,
            bestScore[(bestK.s1 + j) % scoreSize].s1,
            bestScore[(bestK.s2 + j) % scoreSize].s2,
            bestScore[(bestK.s3 + j) % scoreSize].s3,
            bestScore[(bestK.s4 + j) % scoreSize].s4,
            bestScore[(bestK.s5 + j) % scoreSize].s5,
            bestScore[(bestK.s6 + j) % scoreSize].s6,
            bestScore[(bestK.s7 + j) % scoreSize].s7,
            bestScore[(bestK.s8 + j) % scoreSize].s8,
            bestScore[(bestK.s9 + j) % scoreSize].s9,
            bestScore[(bestK.sA + j) % scoreSize].sA,
            bestScore[(bestK.sB + j) % scoreSize].sB,
            bestScore[(bestK.sC + j) % scoreSize].sC,
            bestScore[(bestK.sD + j) % scoreSize].sD,
            bestScore[(bestK.sE + j) % scoreSize].sE,
            bestScore[(bestK.sF + j) % scoreSize].sF
        );
        mass += m;
        sumX += m * jf;
    }
    disparity = convert_float16(bestI - scoreSize) + (sumX / mass);

    return convert_float16(bestI);
}

float3 getDisparityV3(
        uchar *data1,
        uchar *data2,
        size_t x,
        size_t y,
        size_t w,
        size_t h,
        int minDisparity,
        int maxDisparity,
        size_t windowSize0,
        int sz,
        float* q,
        bool debug
) {
    const uchar *src = data1 + y * w * sz + x;
    const uchar *dest = data2 + y * w * sz + x;

    float3 disparity;
    float3 avgScore;
    float3 maxScore;
    maxDisparity = max(minDisparity, maxDisparity);
    minDisparity = min(minDisparity, maxDisparity);
    int disparityRange = maxDisparity - minDisparity;

    const int maxScoreSize = 5;
    int scoreSize = max(2, min(maxScoreSize, disparityRange / 6));
    float3 score[maxScoreSize];
    float3 bestScore[maxScoreSize];

    for (int i = 0; i < scoreSize; ++i) {
        bestScore[i] = 0;
        score[i] = 0;
    }

    int3 bestI = 0;
    int3 bestK = 0;

    int wi = 0;
    int wis[] = {1};
    int wstep = 1;
    int k = 0;
//    do {
        int windowSize = (int)windowSize0 * wis[wi];
        float maxPossibleScore = 255.f * 255.f * (float)windowSize * (float)h;
        maxScore = (float3)0;
        avgScore = (float3)0;

        k = 0;
        float3 scoreSum = 0;

        for (int i = minDisparity; i < maxDisparity; i += sz) {
            float3 score0 = (float3)0;

            int hstep = w * sz;
            for (int j = 0; j < h * hstep; j+= hstep) {
                for (int i0 = 0; i0 <= windowSize / 4 + 1; i0+=wstep) {
                    const float3 d0 = convert_float3(vload3(i0, src + j));
                    const float3 d1 = convert_float3(vload3(i0, dest + i + j));
                    const float3 d = d0 - d1;
                    score0 += d * d;
                }
            }

            float3 newScore = (maxPossibleScore - score0) / (float)windowSize;
            float3 prevScore = score[k % scoreSize];
            score[k % scoreSize] = newScore;

            scoreSum += newScore - prevScore;

            int n = min(k + 1, (int) scoreSize);

            float3 currentScore = scoreSum / (float)n;

            avgScore += newScore;

            int3 needUpdate = isgreater(currentScore, maxScore);

            bestI = select(bestI, (int3)i / sz, needUpdate);
            bestK = select(bestK, (int3)k, needUpdate);
            maxScore = fmax(maxScore, currentScore);

            for (int j = 0; j < scoreSize; ++j) {
                bestScore[j] = select(bestScore[j], score[j], needUpdate);
            }

            k++;
        }
//
//        avgScore /= (float)k;
//        wi++;
//    } while (wi < sizeof wis / sizeof *wis && avgScore > maxScore * 0.9);

//    if (avgScore > maxScore * *q) {
//        *q = *q + 3e-7;
//        return 0;
//    }
//
//    *q = *q - 1e-7;

    scoreSize = min(k, scoreSize);
    float3 mass = 0;
    float3 sumX = 0;

    float jf = 1;
    for (int j = 1; j <= scoreSize; ++j, ++jf) {
        float3 m = (float3)(
            bestScore[(bestK.s0 + j) % scoreSize].s0,
            bestScore[(bestK.s1 + j) % scoreSize].s1,
            bestScore[(bestK.s2 + j) % scoreSize].s2
        );
        mass += m;
        sumX += m * jf;
    }
    disparity = convert_float3(bestI - scoreSize) + (sumX / mass);

    return disparity;
}

__kernel void DisparityEvaluator(
        __global uchar* frame0,
        __global uchar* frame1,
        __global float* roughDisparity,
        __global short* disparity,
        __global float* q,
        const float q0,
        const int windowHeight,
        const int windowSize,
        const int w,
        const int h,
        const int sz,
        const int DISPARITY_PRECISION
) {
    int x0 = get_local_id(0) * 16;
    int y0 = get_global_id(1) * 2;

    float _q = q0;
    int wsz = w * sz;

    int windowSize0 = 3;

#define maxFragmentHeight 3
#define maxFragmentWidth (256 + 30) * 3
#define maxBuffer maxFragmentWidth * maxFragmentHeight
#define maxRowWidth 1920 * 3
#define maxLocalBuffer maxRowWidth * (maxFragmentHeight + 1)

    int fragmentHeight = min(maxFragmentHeight, windowHeight);

    __local uchar pFrame0[maxLocalBuffer];
    __local uchar pFrame1[maxLocalBuffer];

    int rowLimit = min(x0 + 16, w);

    for (int y = y0; y < y0 + 2; ++y) {
        for (int i = 0; i < fragmentHeight; ++i) {
            for (int x = x0; x < rowLimit; x++) {
                for (int ch = 0; ch < 4; ch++) {
    //                pFrame0[i * wsz + x] = frame0[(y + i) * wsz + x];
    //                pFrame1[i * wsz + x] = frame1[(y + i) * wsz + x];
                    pFrame0[i * 4 + ch + x * 4 * fragmentHeight] = frame0[(y + i) * wsz + x * sz + ch];
                    pFrame1[i * 4 + ch + x * 4 * fragmentHeight] = frame1[(y + i) * wsz + x * sz + ch];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int x = 0; x < rowLimit - x0; x += 16) {
//        for (int x = 0; x < rowLimit - x0; x++) {
            int maxDisparity = min(256 * sz * fragmentHeight, (int) (wsz - windowSize - (x + x0) * sz) * fragmentHeight);
//            float d = getDisparity(pFrame1, pFrame0, (x + x0) * sz * fragmentHeight, 0, w, 1, 0, maxDisparity, windowSize * fragmentHeight, sz * fragmentHeight, &_q);
            float16 d = getDisparityV16(pFrame1, pFrame0, (x + x0) * sz * fragmentHeight, 0, w, 1, 0, maxDisparity, windowSize * fragmentHeight, 4 * fragmentHeight, &_q, x0 == 32 && y == 42);

            d *= (float) DISPARITY_PRECISION;

//            disparity[y * w + x + x0] = (short) rint(d);
            disparity[y * w + x + x0 + 0x0] = (short) rint(d.s0);
            disparity[y * w + x + x0 + 0x1] = (short) rint(d.s1);
            disparity[y * w + x + x0 + 0x2] = (short) rint(d.s2);
            disparity[y * w + x + x0 + 0x3] = (short) rint(d.s3);
            disparity[y * w + x + x0 + 0x4] = (short) rint(d.s4);
            disparity[y * w + x + x0 + 0x5] = (short) rint(d.s5);
            disparity[y * w + x + x0 + 0x6] = (short) rint(d.s6);
            disparity[y * w + x + x0 + 0x7] = (short) rint(d.s7);
            disparity[y * w + x + x0 + 0x8] = (short) rint(d.s8);
            disparity[y * w + x + x0 + 0x9] = (short) rint(d.s9);
            disparity[y * w + x + x0 + 0xA] = (short) rint(d.sA);
            disparity[y * w + x + x0 + 0xB] = (short) rint(d.sB);
            disparity[y * w + x + x0 + 0xC] = (short) rint(d.sC);
            disparity[y * w + x + x0 + 0xD] = (short) rint(d.sD);
            disparity[y * w + x + x0 + 0xE] = (short) rint(d.sE);
            disparity[y * w + x + x0 + 0xF] = (short) rint(d.sF);

        }
    }

    *q += _q - q0;
}
