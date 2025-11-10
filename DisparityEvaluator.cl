#include "opencl_ide_defs.h"

#define EACH2(expr) expr.s0; expr.s1;
#define EACH3(expr) EACH2(expr) expr.s2;
#define EACH4(expr) EACH3(expr) expr.s3;
#define EACH5(expr) EACH4(expr) expr.s4;
#define EACH6(expr) EACH5(expr) expr.s5;
#define EACH7(expr) EACH6(expr) expr.s6;
#define EACH8(expr) EACH7(expr) expr.s7;
#define EACH9(expr) EACH8(expr) expr.s8;
#define EACH10(expr) EACH9(expr) expr.s9;
#define EACH11(expr) EACH10(expr) expr.sA;
#define EACH12(expr) EACH11(expr) expr.sB;
#define EACH13(expr) EACH12(expr) expr.sC;
#define EACH14(expr) EACH13(expr) expr.sD;
#define EACH15(expr) EACH14(expr) expr.sE;
#define EACH16(expr) EACH15(expr) expr.sF;

#ifndef GRANULE_SIZE
#define GRANULE_SIZE 2
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

#if DEBUG == 1
struct __bounds_checker {
    char t;
    uintptr_t ptr;
    uintptr_t start;
    size_t end;
    size_t line;
    size_t n;
};

#define _CHECK_BOUNDARY(type, typeId, _ptr, dataStart, dataSize) \
    if (((type char *)(_ptr) < ((type char *)(dataStart)) || (type char *)(_ptr) >= (((type char *)(dataStart)) + (size_t)(dataSize)))) {\
        __bc->n++; __bc->t = typeId; __bc->ptr = (uintptr_t)(_ptr); __bc->start = (uintptr_t)(dataStart); __bc->end = (uintptr_t)(dataStart) + (size_t)(dataSize); __bc->line = __LINE__;}
#define _FATAL_BOUNDARY(type, typeId, _ptr, dataStart, dataSize, result) \
    if (((type char *)(_ptr) < ((type char *)(dataStart)) || (type char *)(_ptr) >= (((type char *)(dataStart)) + (size_t)(dataSize)))) {\
        __bc->n++; __bc->t = typeId; __bc->ptr = (uintptr_t)(_ptr); __bc->start = (uintptr_t)(dataStart); __bc->end = (uintptr_t)(dataStart) + (size_t)(dataSize); __bc->line = __LINE__;return result;}

#define CHECK_GLOBAL_BOUNDARY(_ptr, dataStart, dataSize) _CHECK_BOUNDARY(__global, 'g', _ptr, dataStart, dataSize)
#define CHECK_LOCAL_BOUNDARY(_ptr, dataStart, dataSize) _CHECK_BOUNDARY(__local, 'l', _ptr, dataStart, dataSize)
#define CHECK_BOUNDARY(_ptr, dataStart, dataSize) _CHECK_BOUNDARY(__private, 'p', _ptr, dataStart, dataSize)

#define FATAL_GLOBAL_BOUNDARY(_ptr, dataStart, dataSize, result) _FATAL_BOUNDARY(__global, 'g', _ptr, dataStart, dataSize, result)
#define FATAL_LOCAL_BOUNDARY(_ptr, dataStart, dataSize, result) _FATAL_BOUNDARY(__local, 'l', _ptr, dataStart, dataSize, result)
#define FATAL_BOUNDARY(_ptr, dataStart, dataSize, result) _FATAL_BOUNDARY(__private, 'p', _ptr, dataStart, dataSize, result)

#define BC_START \
    __private struct __bounds_checker __bc0 = {0,0,0,0}; \
    __private struct __bounds_checker *__bc = &__bc0;

#define BC_PASS , __bc
#define BC_ARG , struct __bounds_checker *__bc
#define BC_DUMP(expr) if (__bc0.n > 0) { expr; printf("ptr %c %p out of range %p-%p %d %d\n", __bc0.t, __bc0.ptr, __bc0.start, __bc0.end, __bc0.line, __bc0.n); }

#else
#define CHECK_GLOBAL_BOUNDARY(_ptr, dataStart, dataSize)
#define CHECK_LOCAL_BOUNDARY(_ptr, dataStart, dataSize)
#define CHECK_BOUNDARY(_ptr, dataStart, dataSize)
#define FATAL_GLOBAL_BOUNDARY(_ptr, dataStart, dataSize, result)
#define FATAL_LOCAL_BOUNDARY(_ptr, dataStart, dataSize, result)
#define FATAL_BOUNDARY(_ptr, dataStart, dataSize, result)
#define BC_START
#define BC_PASS
#define BC_ARG
#define BC_DUMP(expr)
#endif

float getDisparity(
        __local char *data1,
        __local char *data2,
        size_t x,
        size_t y,
        size_t w,
        size_t h,
        int minDisparity,
        int maxDisparity,
        size_t windowSize0,
        int sz,
        float* q
        BC_ARG
) {
    __local const char *src = data1 + y * w * sz + x;
    __local const char *dest = data2 + y * w * sz + x;

    size_t dataSize = w * sz * h;

    float disparity;
    float avgScore;
    float maxScore;
    maxDisparity = max(minDisparity, maxDisparity);
    minDisparity = min(minDisparity, maxDisparity);
    int disparityRange = maxDisparity - minDisparity;

    CHECK_LOCAL_BOUNDARY(&src[(h - 1) * w * sz + windowSize0], data1, dataSize);
    CHECK_LOCAL_BOUNDARY(&dest[maxDisparity + (h - 1) * w * sz + windowSize0], data2, dataSize);

    CHECK_LOCAL_BOUNDARY(src, data1, dataSize);
    CHECK_LOCAL_BOUNDARY(dest, data2, dataSize);
    CHECK_LOCAL_BOUNDARY(&dest[minDisparity], data2, dataSize);

    const int maxScoreSize = 7;
    uint scoreSize = max(2, min(maxScoreSize, disparityRange / sz / 6));
    __private float score[maxScoreSize];
    __private float bestScore[maxScoreSize];

    for (int i = 0; i < scoreSize; ++i) {
        bestScore[i] = 0;
        score[i] = 0;
    }

    int bestI = 0;
    uint bestK = 0;

    int wi = 0;
    int wis[] = {1};
    int wstep = 1;
    uint k = 0;

    do {
        int windowSize = (int)windowSize0 * wis[wi];
        float maxPossibleScore = 255.f * 255.f * (float)windowSize * (float)h;
        maxScore = 0;
        avgScore = 0;

        k = 0;
        float scoreSum = 0;

        for (int i = minDisparity; i <= maxDisparity; i += sz) {
            float score0 = 0;

            int hstep = w * sz;
            for (int j = 0; j < h * hstep; j += hstep) {
                const int windowSize16 = windowSize / 16;
                const int windowSize8 = windowSize / 8;
                const int windowSize4 = windowSize / 4;
                const int windowSize2 = windowSize / 2;
                for (int i0 = 0; i0 < windowSize16; i0+=1) {
                    const char16 d0 = vload16(i0, &src[j]) - vload16(i0, &dest[i + j]);
                    float16 df = convert_float16(d0);
                    df *= df;
                    EACH16(score0 += df)
                }
                for (int i0 = windowSize16 * 16; i0 < windowSize8; i0+=1) {
                    const char8 d0 = vload8(i0, &src[j]) - vload8(i0, &dest[i + j]);
                    float8 df = convert_float8(d0);
                    df *= df;
                    EACH8(score0 += df)
                }
                for (int i0 = windowSize8 * 8; i0 <= windowSize; i0+=1) {
                    const char d0 = src[j + i0] - dest[i + j + i0];
                    const float df = (float)d0;
                    score0 += df * df;
                }
            }

            float newScore = (maxPossibleScore - score0) / (float)windowSize;
            float prevScore = score[k % scoreSize];

            score[k % scoreSize] = newScore;

            scoreSum += newScore - prevScore;

            uint n = min(k + 1, scoreSize);

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
        wstep++;
    } while (false);
//
//    if (avgScore > maxScore * *q) {
//        *q = *q + 3e-7;
//        return 0;
//    }
//
//    *q = *q - 1e-7;

    scoreSize = min(k + 1, scoreSize);
    int n = scoreSize;
    float mass = 0;
    float sumX = 0;

    float min = 1e12;

    for (int i = 0; i < n; ++i) {
        min = fmin(min, bestScore[i]);
    }
    float f = 1;
    for (uint i = 1; i <= n; ++i, ++f) {
        CHECK_BOUNDARY(&bestScore[(bestK + i) % n], bestScore, maxScoreSize * sizeof *bestScore);
        float m = bestScore[(bestK + i) % n] - min;
        mass += m;
        sumX += m * f;
    }

    disparity = (float)(bestI - n) + (sumX / mass);

    return disparity;
}

#define maxFragmentHeight 3
#define nsz 3
#define maxRowWidth 1920 * nsz
#define maxLocalBuffer maxRowWidth * (maxFragmentHeight)
#define MIN_VALID_DISPARITY 5

__kernel void DisparityEvaluator(
        __global char* frame0,
        __global char* frame1,
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
    BC_START;

    int x0 = get_local_id(0) * GRANULE_SIZE;
    int y0 = get_global_id(1) * GRANULE_SIZE;

    float _q = q0;
    int wsz = w * sz;

    int windowSize0 = 3;

    __local char pFrame0[maxLocalBuffer];
    __local char pFrame1[maxLocalBuffer];
    __local uchar safety[1000];

    int xLimit = min(x0 + GRANULE_SIZE, w - 1);
    int yLimit = min(y0 + GRANULE_SIZE, h - 1);

    int fragmentHeight = min(maxFragmentHeight, windowHeight);
    fragmentHeight = min(fragmentHeight, h - 1 - y0);

    bool debug = true;

    for (int y = y0; y < yLimit; ++y) {
        for (int i = 0; i < fragmentHeight; ++i) {
            for (int x = x0; x < xLimit; x++) {
                for (int ch = 0; ch < sz; ch++) {
                    CHECK_LOCAL_BOUNDARY(&pFrame0[i * nsz + ch + x * nsz * fragmentHeight], pFrame0, maxLocalBuffer);
                    CHECK_LOCAL_BOUNDARY(&pFrame1[i * nsz + ch + x * nsz * fragmentHeight], pFrame1, maxLocalBuffer);
                    CHECK_GLOBAL_BOUNDARY(&frame0[(y + i) * wsz + x * sz + ch], frame0, wsz * h);
                    CHECK_GLOBAL_BOUNDARY(&frame1[(y + i) * wsz + x * sz + ch], frame1, wsz * h);
                    pFrame0[i * nsz + ch + x * nsz * fragmentHeight] = frame0[(y + i) * wsz + x * sz + ch];
                    pFrame1[i * nsz + ch + x * nsz * fragmentHeight] = frame1[(y + i) * wsz + x * sz + ch];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int x = x0; x < min(xLimit, w - MIN_VALID_DISPARITY); x += 1) {
            int maxDisparity = max(0, min(256 * nsz, (int) (w * nsz - ((x + 1) * nsz + windowSize + nsz)))) * fragmentHeight;
            float d0 = getDisparity(pFrame1, pFrame0, x * nsz * fragmentHeight, 0, w, 1, 0, maxDisparity, windowSize * fragmentHeight, nsz * fragmentHeight, &_q BC_PASS);
            short id0 = (int) rint(d0);
            float xd0 = w * nsz - (x + id0) * nsz < windowSize + nsz
                    ? d0
                    : -getDisparity(pFrame0, pFrame1, (x + id0) * nsz * fragmentHeight, 0, w, 1, -min((x + id0) * nsz * fragmentHeight, 256 * nsz * fragmentHeight), 0, windowSize * fragmentHeight, nsz * fragmentHeight, &_q BC_PASS);
            float d = d0 * (float) DISPARITY_PRECISION;
            float xd = xd0 * (float) DISPARITY_PRECISION;
            short sxd = (short) rint(xd);

            bool valid = fabs(d0 - xd0) / d0 < 0.1;

            disparity[y * w + x] = sxd;

        }
    }

    BC_DUMP();

    *q += _q - q0;
}
