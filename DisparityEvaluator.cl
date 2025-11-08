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
    do {
        int windowSize = (int)windowSize0 * wis[wi];
        float maxPossibleScore = 255.f * 255.f * (float)windowSize * (float)h;
        maxScore = 0;
        disparity = 0;
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

        avgScore /= (float)k;
        wi++;
    } while (wi < sizeof wis / sizeof *wis && avgScore > maxScore * 0.9);

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
            for (int x = x0 * sz; x < rowLimit * sz; ++x) {
                pFrame0[i * wsz + x] = frame0[(y + i) * wsz + x];
                pFrame1[i * wsz + x] = frame1[(y + i) * wsz + x];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int x = 0; x < rowLimit - x0; ++x) {
            int maxDisparity = min(256 * sz, (int) (wsz - windowSize - (x + x0) * sz));
            float d = getDisparity(pFrame1, pFrame0, (x + x0) * sz, 0, w, fragmentHeight, 0, maxDisparity, windowSize, sz, &_q);

            disparity[y * w + x + x0] = (short) rint(d * (float) DISPARITY_PRECISION);
        }
    }

    *q += _q - q0;
}
