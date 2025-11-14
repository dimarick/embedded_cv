#include "opencl_ide_defs.h"

#ifndef H_GRANULE_SIZE
#define H_GRANULE_SIZE 8
#endif

#ifndef V_GRANULE_SIZE
#define V_GRANULE_SIZE 2
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

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
#define printf(...)
#endif

#define VECTOR_FOR_START(dim, i, rangeStart, rangeEnd) \
{ int __nextVal = (rangeStart); int __rangeStart = (rangeStart); int __rangeEnd = (rangeEnd); \
for (int i = (rangeStart); i < (rangeStart) + ((rangeEnd) - (rangeStart)) / dim; i++, __nextVal += dim) {

#define VECTOR_FOR_CONTINUE(dim, i) }; for (int i = __nextVal; i < __rangeStart + (__rangeEnd - __rangeStart) / dim; i++) {

#define VECTOR_FOR_TAIL(i) } for (int i = __nextVal; i < __rangeEnd; i++) {

#define VECTOR_FOR_END }}


float8 getDisparity(
        __local char *data1,
        __local char *data2,
        int x,
        int y,
        int w,
        int h,
        int8 xDelta,
        int minDisparity,
        int maxDisparity,
        int windowSize0,
        int sz,
        float* q,
        bool debug
        BC_ARG
) {
    __local const char *src = data1 + y * sz + x * h * sz;
    __local const char *dest = data2 + y * sz + x * h * sz;

    size_t dataSize = w * sz * h;

    float8 disparity;
    float avgScore;
    maxDisparity = max(minDisparity, maxDisparity);
    minDisparity = min(minDisparity, maxDisparity);
    int disparityRange = maxDisparity - minDisparity;

    CHECK_LOCAL_BOUNDARY(src, data1, dataSize);
    CHECK_LOCAL_BOUNDARY(dest, data2, dataSize);
    CHECK_LOCAL_BOUNDARY(&dest[minDisparity], data2, dataSize);

    const int maxScoreSize = 3;
    const int scoreSize = max(1, min(maxScoreSize, disparityRange / 6));
    float8 score[maxScoreSize];
    float8 bestScore[maxScoreSize];

    CHECK_BOUNDARY(&bestScore[scoreSize-1], bestScore, maxScoreSize * sizeof *bestScore);
    CHECK_BOUNDARY(&score[scoreSize-1], score, maxScoreSize * sizeof *score);
    for (int i = 0; i < scoreSize; ++i) {
        bestScore[i] = (float8)0;
        score[i] = (float8)0;
    }

    int8 bestD = (int8)0;

    float8 scoreSum = (float8)0;
    int currentScoreSize = 1;

    int xDeltaArray[8];
    vstore8(xDelta, 0, xDeltaArray);

    float8 maxScore = (float8)-1e12;
    for (int d = minDisparity; d <= maxDisparity; d++) {
        // предположим что окно соответствия равно 4*4, а maxDisparity-minDisparity кратно 16
        // а формат входных данных: src[tileNo][x1..xn][y1..y4][ch]
        // в score0 поместим стоимость первой колонки пикселей для x
        float score0[8];
        // в score0 поместим стоимость всего тайла от 0 до N-1 = scoreX0 = score0 + score1 + .. + scoreN
        float score16 = 0;
        // тогда score x1 scoreX1 = score1 + .. + scoreN+1 = score - score0 + scoreN+1

        const char16 d0 = vload16(0, &src[xDelta.s0 * h * sz]) - vload16(0, &dest[d * h * sz + xDelta.s0 * h * sz]);
        const char16 d1 = vload16(1, &src[xDelta.s0 * h * sz]) - vload16(1, &dest[d * h * sz + xDelta.s0 * h * sz]);
        const char16 d2 = vload16(2, &src[xDelta.s0 * h * sz]) - vload16(2, &dest[d * h * sz + xDelta.s0 * h * sz]);
        float16 df0 = convert_float16(d0);
        float16 df1 = convert_float16(d1);
        float16 df2 = convert_float16(d2);
        df0 *= df0;
        df1 *= df1;
        df2 *= df2;

        score0[0x0] = df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4 + df0.s5 + df0.s6 + df0.s7 + df0.s8 + df0.s9 + df0.sA + df0.sB;
        score0[0x1] = df0.sC + df0.sD + df0.sE + df0.sF + df1.s0 + df1.s1 + df1.s2 + df1.s3 + df1.s4 + df1.s5 + df1.s6 + df1.s7;
        score0[0x2] = df1.s8 + df1.s9 + df1.sA + df1.sB + df1.sC + df1.sD + df1.sE + df1.sF + df2.s0 + df2.s1 + df2.s2 + df2.s3;
        score0[0x3] = df2.s4 + df2.s5 + df2.s6 + df2.s7 + df2.s8 + df2.s9 + df2.sA + df2.sB + df2.sC + df2.sD + df2.sE + df2.sF;

        EACH16(score16 += df0)
        EACH16(score16 += df1)
        EACH16(score16 += df2)
//        }

        float scores[8];
        vstore8((float8)0, 0, scores);
        scores[0] = score16;

        for (int xi = 1; xi < 8; xi++) {
            float scoreN = 0;
            for (int yi = 0; yi < h; yi++) {
                const char3 c0 = vload3((xi + 4 + xDeltaArray[xi]) * h + yi, &src[0]) - vload3((xi + 4 + d) * h + yi, &dest[0]);
                float3 cf = convert_float3(c0);
                cf *= cf;
                EACH3(scoreN += cf)
            }
            score16 += scoreN - score0[(xi - 1) % 8];
            score0[(xi + 3) % 8] = scoreN;
            scores[xi] = score16;
        }

        float8 vScores = vload8(0, scores);
        float8 newScore = vScores / (float)-windowSize0;

        int scoreOffset = (d % scoreSize + scoreSize) % scoreSize;
        float8 prevScore = score[scoreOffset];
        score[scoreOffset] = newScore;

        scoreSum += newScore - prevScore;

        float8 currentScore = scoreSum / (float8)currentScoreSize;

//            avgScore += newScore;

        int8 needUpdate = isgreater(currentScore, maxScore);
        bestD = select(bestD, (int8)d - xDelta, needUpdate);
        maxScore = fmax(maxScore, currentScore);

        for (int i = 0; i < scoreSize; i++) {
            bestScore[i] = select(bestScore[i], score[i], needUpdate);
        }

//        if (debug) printf("%d %d %d %d (%d, %d, %d, %d) (%d, %d, %d, %d)\n", scoreOffset, d, minDisparity, x, xDelta.s0, xDelta.s1, xDelta.s2, xDelta.s3, needUpdate.s0, needUpdate.s1, needUpdate.s2, needUpdate.s3);

        currentScoreSize = min(currentScoreSize + 1, scoreSize);
    }

    float8 mass = 0;
    float8 sumX = 0;

    float8 min = 1e12;

    for (int i = 0; i < scoreSize; i++) {
        min = fmin(min, bestScore[i]);
    }

    float f = 1;

    for (int i = 0; i < scoreSize; ++i) {
        float8 x0 = convert_float8(((bestD + i) % scoreSize + scoreSize) % scoreSize);
        float8 m = bestScore[i] - (float8)min;
        mass += m;
        sumX += x0 * m;
    }

    float8 massSafe = select(mass, (float8)1, isequal(mass, (float8) 0));
    float8 sumXSafe = select(sumX, convert_float8(scoreSize), isequal(mass, (float8) 0));
    disparity = convert_float8(bestD - scoreSize / 2) + sumXSafe / massSafe;

    if (debug) printf("%d %d (%f,%f,%f,%f, %f,%f,%f,%f) %f %f\n", minDisparity, x, disparity.s0, disparity.s1, disparity.s2, disparity.s3, disparity.s4, disparity.s5, disparity.s6, disparity.s7, bestScore[0].s0, score[0].s0);

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



//        for (int t = x0 / tileW; t < fragmentHeight * H_GRANULE_SIZE / tileW; t++) {
//            int yi = t * tileSize / w;
//            for (int yj = 0; yj < tileH; ++yj) {
//                for (int x = 0; x < tileW; ++x) {
//                    for (int ch = 0; ch < sz; ch++) {
//                        pFrame0[t * tileSize * nsz + x * tileH * nsz + yj * nsz + ch] = frame0[(y + yi + yj) * wsz + (x0 + x) * sz + ch];
//                        pFrame1[t * tileSize * nsz + x * tileH * nsz + yj * nsz + ch] = frame1[(y + yi + yj) * wsz + (x0 + x) * sz + ch];
//                    }
//                }
//            }
//        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int x = x0; x < min(xLimit, w - MIN_VALID_DISPARITY - windowSize0); x += 8) {
            const int maxDisparity = max(0, min(256, (int) (w - ((x + 1) + windowSize0 + 1))));
            const float8 d0 = getDisparity(pFrame1, pFrame0, x, 0, w, fragmentHeight, (int8)0, 0, maxDisparity, windowSize0, nsz, &_q, x0==960 && y0 == 540 BC_PASS);
            const int8 id0 = convert_int8(rint(d0));

            const int minDisparity = max(-x, -256);
            const int maxDisparityX = 0;
            const float8 xd0 = w - (x + id0.s0) < windowSize0 + 1
                    ? d0
                    : -getDisparity(pFrame0, pFrame1, x + id0.s0, 0, w, fragmentHeight, id0 - (int8)id0.s0, minDisparity, maxDisparityX, windowSize0, nsz, &_q, x0==320 && y0 == 640 BC_PASS);
            const float8 xd = fmin(xd0, d0) * (float) DISPARITY_PRECISION;
            const short8 sxd = convert_short8(rint(xd));
            const short8 sd = convert_short8(rint(d0 * (float) DISPARITY_PRECISION));

            vstore8(sxd, 0, &disparity[y * w + x]);
//            disparity[y * w + x] = sxd.s0;
//            disparity[y * w + x + 1] = sxd.s1;
//            disparity[y * w + x + 2] = sxd.s2;
//            disparity[y * w + x + 3] = sxd.s3;
        }
    }

    BC_DUMP("(%d;%d)", x0, y0);
//
//    *q += _q - q0;
}
