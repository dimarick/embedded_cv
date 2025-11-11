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
#define GRANULE_SIZE 4
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
#define BC_DUMP(expr)
#define CHECK_NUMBER(v)
#define printf
#endif

#define VECTOR_FOR_START(dim, i, rangeStart, rangeEnd) \
{ int __nextVal = (rangeStart); \
for (int i = (rangeStart); i < (rangeStart) + ((rangeEnd) - (rangeStart)) / dim; i++, __nextVal += dim) {

#define VECTOR_FOR_CONTINUE(dim, i, rangeStart, rangeEnd) }; for (int i = __nextVal; i < (rangeStart) + ((rangeEnd) - (rangeStart)) / dim; i++) {

#define VECTOR_FOR_TAIL(i, rangeStart, rangeEnd) } for (int i = __nextVal; i < (rangeEnd); i++) {

#define VECTOR_FOR_END }}


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
        float* q,
        bool debug
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

    const uint maxScoreSize = 8;
    uint scoreSize = max((uint)2, min(maxScoreSize, (uint)disparityRange / sz / 6));
    __private float score[maxScoreSize];
    __private float bestScore[maxScoreSize];

    CHECK_BOUNDARY(&bestScore[scoreSize-1], bestScore, maxScoreSize * sizeof *bestScore);
    CHECK_BOUNDARY(&score[scoreSize-1], score, maxScoreSize * sizeof *score);
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
                VECTOR_FOR_START(16, i0, 0, windowSize)
                    const char16 d0 = vload16(i0, &src[j]) - vload16(i0, &dest[i + j]);
                    float16 df = convert_float16(d0);
                    df *= df;
                    EACH16(score0 += df)
                VECTOR_FOR_CONTINUE(8, i0, 0, windowSize)
                    const char8 d0 = vload8(i0, &src[j]) - vload8(i0, &dest[i + j]);
                    float8 df = convert_float8(d0);
                    df *= df;
                    EACH8(score0 += df)
                VECTOR_FOR_CONTINUE(4, i0, 0, windowSize)
                    const char4 d0 = vload4(i0, &src[j]) - vload4(i0, &dest[i + j]);
                    float4 df = convert_float4(d0);
                    df *= df;
                    EACH4(score0 += df)
                VECTOR_FOR_CONTINUE(2, i0, 0, windowSize)
                    const char2 d0 = vload2(i0, &src[j]) - vload2(i0, &dest[i + j]);
                    float2 df = convert_float2(d0);
                    df *= df;
                    EACH2(score0 += df)
                VECTOR_FOR_TAIL(i0, 0, windowSize)
                    const char d0 = src[j + i0] - dest[i + j + i0];
                    const float df = (float)d0;
                    score0 += df * df;
                VECTOR_FOR_END
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

            const int copySize = (int)needUpdate * scoreSize;

            VECTOR_FOR_START(8, j, 0, copySize)
                vstore8(vload8(j, &score[j]), j, &bestScore[j]);
            VECTOR_FOR_CONTINUE(4, j, 0, copySize)
                vstore4(vload4(j, &score[j]), j, &bestScore[j]);
            VECTOR_FOR_CONTINUE(2, j, 0, copySize)
                vstore2(vload2(j, &score[j]), j, &bestScore[j]);
            VECTOR_FOR_TAIL(j, 0, copySize)
                bestScore[j] = score[j];
            VECTOR_FOR_END

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

    VECTOR_FOR_START(8, i, 0, n)
        const float8 v = vload8(i, bestScore);
        min = fmin(min, v.s0);
        min = fmin(min, v.s1);
        min = fmin(min, v.s2);
        min = fmin(min, v.s3);
        min = fmin(min, v.s4);
        min = fmin(min, v.s5);
        min = fmin(min, v.s6);
        min = fmin(min, v.s7);
    VECTOR_FOR_TAIL(i, 1, n + 1)
        min = fmin(min, bestScore[i]);
    VECTOR_FOR_END

    float f = 1;
    VECTOR_FOR_START(8, i, 1, n + 1)
        float8 m = vload8(i, &bestScore[(bestK + i) % n]) - min;
        EACH8(mass += m);
        EACH8(sumX += (f++) * m);
    VECTOR_FOR_CONTINUE(4, i, 1, n + 1)
        float4 m = vload4(i, &bestScore[(bestK + i) % n]) - min;
        EACH4(mass += m);
        EACH4(sumX += (f++) * m);
    VECTOR_FOR_CONTINUE(2, i, 1, n + 1)
        float2 m = vload2(i, &bestScore[(bestK + i) % n]) - min;
        EACH2(mass += m);
        EACH2(sumX += (f++) * m);
    VECTOR_FOR_TAIL(i, 1, n + 1)
        float m = bestScore[(bestK + i) % n] - min;
        mass += m;
        sumX += m * f;
    VECTOR_FOR_END

    disparity = (sumX > 0 && mass > 0) ? ((float)(bestI - n) + (sumX / mass)) : (float)bestI;

    return fmax(minDisparity / sz, fmin(disparity, maxDisparity / sz));
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

    int x0 = get_local_id(0) * GRANULE_SIZE;
    int y0 = get_global_id(1) * GRANULE_SIZE;

    float _q = q0;
    int wsz = w * sz;

    int windowSize0 = 4;

    __local long upFrame0[maxLocalBuffer / 8 + 1];
    __local long upFrame1[maxLocalBuffer / 8 + 1];

    __local char *pFrame0 = (__local char *)upFrame0;
    __local char *pFrame1 = (__local char *)upFrame1;

    int xLimit = min(x0 + GRANULE_SIZE, w - 1);
    int yLimit = min(y0 + GRANULE_SIZE, h - 1);

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

        for (int x = x0; x < min(xLimit, w - MIN_VALID_DISPARITY); x += 1) {
            int maxDisparity = max(0, min(384 * nsz, (int) (w * nsz - ((x + 1) * nsz + windowSize0 + nsz)))) * fragmentHeight;
            float d0 = getDisparity(pFrame1, pFrame0, x * nsz * fragmentHeight, 0, w, 1, 0, maxDisparity, windowSize0 * fragmentHeight, nsz * fragmentHeight, &_q, x0==30 && y0 == 40 BC_PASS);
            short id0 = (short) rint(d0);
            float xd0 = w * nsz - (x + id0) * nsz < windowSize0 + nsz
                    ? d0
                    : -getDisparity(pFrame0, pFrame1, (x + id0) * nsz * fragmentHeight, 0, w, 1, -min((x + id0) * nsz * fragmentHeight, 256 * nsz * fragmentHeight), 0, windowSize0 * fragmentHeight, nsz * fragmentHeight, &_q, x0==30 && y0 == 40 BC_PASS);
            float d = d0 * (float) DISPARITY_PRECISION;
            float xd = xd0 * (float) DISPARITY_PRECISION;
            short sxd = (short) rint(xd);

            disparity[y * w + x] = sxd;

        }
    }

    BC_DUMP();

    *q += _q - q0;
}
