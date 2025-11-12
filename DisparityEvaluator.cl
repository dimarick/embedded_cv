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

#ifndef H_GRANULE_SIZE
#define H_GRANULE_SIZE 8
#endif

#ifndef V_GRANULE_SIZE
#define V_GRANULE_SIZE 2
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
#define BC_DUMP(expr, ...) if (__bc0.n > 0) { printf(expr ". Ptr %c %p out of range %p-%p %d %d\n", __VA_ARGS__, __bc0.t, __bc0.ptr, __bc0.start, __bc0.end, __bc0.line, __bc0.n); }

#define CHECK_NUMBER(v) \
    if (isinf(v)) { __bc->n++; __bc->t = 'f'; __bc->line = __LINE__; } \
    if (isnan(v)) { __bc->n++; __bc->t = 'n'; __bc->line = __LINE__; }

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
#define BC_DUMP(expr, ...)
#define CHECK_NUMBER(v)
#define printf
#endif

#define VECTOR_FOR_START(dim, i, rangeStart, rangeEnd) \
{ int __nextVal = (rangeStart); int __rangeStart = (rangeStart); int __rangeEnd = (rangeEnd); \
for (int i = (rangeStart); i < (rangeStart) + ((rangeEnd) - (rangeStart)) / dim; i++, __nextVal += dim) {

#define VECTOR_FOR_CONTINUE(dim, i) }; for (int i = __nextVal; i < __rangeStart + (__rangeEnd - __rangeStart) / dim; i++) {

#define VECTOR_FOR_TAIL(i) } for (int i = __nextVal; i < __rangeEnd; i++) {

#define VECTOR_FOR_END }}


float getDisparity(
        __local char *data1,
        __local char *data2,
        int x,
        int y,
        int w,
        int h,
        int minDisparity,
        int maxDisparity,
        int windowSize0,
        int sz,
        float* q,
        bool debug
        BC_ARG
) {
    __local const char *src = data1 + y * w * sz + x * sz;
    __local const char *dest = data2 + y * w * sz + x * sz;

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

    const int maxScoreSize = 8;
    const int scoreSize = max(2, min(maxScoreSize, disparityRange / sz / 6));
    float score[maxScoreSize];
    float bestScore[maxScoreSize];

    CHECK_BOUNDARY(&bestScore[scoreSize-1], bestScore, maxScoreSize * sizeof *bestScore);
    CHECK_BOUNDARY(&score[scoreSize-1], score, maxScoreSize * sizeof *score);
    for (int i = 0; i < scoreSize; ++i) {
        bestScore[i] = 0;
        score[i] = 0;
    }

    int bestI = 0;

    int wi = 0;
    int wis[] = {1};
    int wstep = 1;
    uint k = 0;

    int windowSize = (int)windowSize0 * wis[wi] * sz;
    float maxPossibleScore = 255.f * 255.f * (float)windowSize * (float)h;
    maxScore = 0;
    avgScore = 0;

    k = 0;
    float scoreSum = 0;

    for (int i = minDisparity; i <= maxDisparity; i++) {
        float score0 = 0;

        int hstep = w * sz;
        for (int j = 0; j < h * hstep; j += hstep) {
            VECTOR_FOR_START(16, i0, 0, windowSize)
                    const char16 d0 = vload16(i0, &src[j]) - vload16(i0, &dest[i * sz + j]);
                    float16 df = convert_float16(d0);
                    df *= df;
                    EACH16(score0 += df)
                VECTOR_FOR_CONTINUE(8, i0)
                    const char8 d0 = vload8(i0, &src[j]) - vload8(i0, &dest[i * sz + j]);
                    float8 df = convert_float8(d0);
                    df *= df;
                    EACH8(score0 += df)
                VECTOR_FOR_CONTINUE(4, i0)
                    const char4 d0 = vload4(i0, &src[j]) - vload4(i0, &dest[i * sz + j]);
                    float4 df = convert_float4(d0);
                    df *= df;
                    EACH4(score0 += df)
                VECTOR_FOR_CONTINUE(2, i0)
                    const char2 d0 = vload2(i0, &src[j]) - vload2(i0, &dest[i * sz + j]);
                    float2 df = convert_float2(d0);
                    df *= df;
                    EACH2(score0 += df)
                VECTOR_FOR_TAIL(i0)
                    const char d0 = src[j + i0] - dest[i * sz + j + i0];
                    const float df = (float) d0;
                    score0 += df * df;
            VECTOR_FOR_END
        }

        float newScore = (maxPossibleScore - score0) / (float) windowSize;
        float prevScore = score[(i % scoreSize + scoreSize) % scoreSize];
        CHECK_BOUNDARY(&score[(i % scoreSize + scoreSize) % scoreSize], score, maxScoreSize * sizeof *score);
        score[(i % scoreSize + scoreSize) % scoreSize] = newScore;

        scoreSum += newScore - prevScore;

        float currentScore = scoreSum / (float) scoreSize;

        avgScore += newScore;

        bool needUpdate = currentScore > maxScore;

        bestI = needUpdate ? i : bestI;
        maxScore = fmax(maxScore, currentScore);

        const int copySize = needUpdate ? scoreSize : 0;

        for (int j = 0; j < copySize; ++j) {
            bestScore[j] = score[j];
        }
    }

    float mass = 0;
    float sumX = 0;

    float min = 1e12;

    for (int i = 0; i < scoreSize; ++i) {
        min = fmin(min, bestScore[i]);
    }

    float f = 1;

    for (int i = 0; i < scoreSize; ++i) {
        float m = bestScore[((bestI + i) % scoreSize + scoreSize) % scoreSize] - min;
        mass += m;
        sumX += m * f++;
    }
    disparity = (sumX > 0 && mass > 0) ? ((float)(bestI - scoreSize) + (sumX / mass)) : (float)bestI;

    return fmax(minDisparity, fmin(disparity, maxDisparity));
}

#define maxFragmentHeight 4
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

    int x0 = get_local_id(0) * H_GRANULE_SIZE;
    int y0 = get_global_id(1) * V_GRANULE_SIZE;

    float _q = q0;
    int wsz = w * sz;

    int windowSize0 = 4;

    __local long upFrame0[maxLocalBuffer / sizeof (long) + 1];
    __local long upFrame1[maxLocalBuffer / sizeof (long) + 1];

    __local char *pFrame0 = (__local char *)upFrame0;
    __local char *pFrame1 = (__local char *)upFrame1;

    int xLimit = min(x0 + H_GRANULE_SIZE, w - 1);
    int yLimit = min(y0 + V_GRANULE_SIZE, h - 1);

    int fragmentHeight = min(maxFragmentHeight, windowHeight);
    fragmentHeight = min(fragmentHeight, h - 1 - y0);

    bool debug = true;

    for (int y = y0; y < yLimit; ++y) {
        for (int i = 0; i < fragmentHeight; ++i) {
            for (int x = x0; x < xLimit; x++) {
                for (int ch = 0; ch < sz; ch++) {
                    pFrame0[i * nsz + ch + x * nsz * fragmentHeight] = frame0[(y + i) * wsz + x * sz + ch];
                    pFrame1[i * nsz + ch + x * nsz * fragmentHeight] = frame1[(y + i) * wsz + x * sz + ch];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int x = x0; x < min(xLimit, w - MIN_VALID_DISPARITY - windowSize0); x += 1) {
            int maxDisparity = max(0, min(256, (int) (w - ((x + 1) + windowSize0 + 1))));
            float d0 = getDisparity(pFrame1, pFrame0, x, 0, w, 1, 0, maxDisparity, windowSize0, nsz * fragmentHeight, &_q, x0==30 && y0 == 40 BC_PASS);
            short id0 = (short) rint(d0);
            float xd0 = w - (x + id0) < windowSize0 + 1
                    ? d0
                    : -getDisparity(pFrame0, pFrame1, x + id0, 0, w, 1, -min(x + id0, 256), 0, windowSize0, nsz * fragmentHeight, &_q, x0==30 && y0 == 40 BC_PASS);
            float d = d0 * (float) DISPARITY_PRECISION;
            float xd = xd0 * (float) DISPARITY_PRECISION;
            short sxd = (short) rint(xd);

            disparity[y * w + x] = sxd;

        }
    }

    BC_DUMP("(%d;%d)", x0, y0);
//
//    *q += _q - q0;
}
