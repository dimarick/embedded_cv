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

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 8
#endif

#define maxFragmentHeight 4
#define nsz 3
#define maxRowWidth 1920 * nsz
#define maxLocalBuffer maxRowWidth * (maxFragmentHeight)
#define MIN_VALID_DISPARITY 5

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

#if VECTOR_SIZE == 16
#define floatN float16
#define intN int16
#define shortN short16
#define charN char16
#define  convert_floatN(...) convert_float16(__VA_ARGS__)
#define  convert_intN(...) convert_int16(__VA_ARGS__)
#define  convert_shortN(...) convert_short16(__VA_ARGS__)
#define  vstoreN(...) vstore16(__VA_ARGS__)
#define  vloadN(...) vload16(__VA_ARGS__)
#elif VECTOR_SIZE == 8
#define floatN float8
#define intN int8
#define shortN short8
#define charN char8
#define  convert_floatN(...) convert_float8(__VA_ARGS__)
#define  convert_intN(...) convert_int8(__VA_ARGS__)
#define  convert_shortN(...) convert_short8(__VA_ARGS__)
#define  vstoreN(...) vstore8(__VA_ARGS__)
#define  vloadN(...) vload8(__VA_ARGS__)
#elif VECTOR_SIZE == 4
#define floatN float4
#define intN int4
#define shortN short4
#define charN char4
#define  convert_floatN(...) convert_float4(__VA_ARGS__)
#define  convert_intN(...) convert_int4(__VA_ARGS__)
#define  convert_shortN(...) convert_short4(__VA_ARGS__)
#define  vstoreN(...) vstore4(__VA_ARGS__)
#define  vloadN(...) vload4(__VA_ARGS__)
#elif VECTOR_SIZE == 2
#define floatN float2
#define intN int2
#define shortN short2
#define charN char2
#define  convert_floatN(...) convert_float2(__VA_ARGS__)
#define  convert_intN(...) convert_int2(__VA_ARGS__)
#define  convert_shortN(...) convert_short2(__VA_ARGS__)
#define  vstoreN(...) vstore2(__VA_ARGS__)
#define  vloadN(...) vload2(__VA_ARGS__)
#elif VECTOR_SIZE == 1
#define floatN float
#define intN int
#define shortN short
#define charN char
#define  convert_floatN(...) (float)(__VA_ARGS__)
#define  convert_intN(...) (int)(__VA_ARGS__)
#define  convert_shortN(...) (short)(__VA_ARGS__)
#define  vstoreN(data, offset, p) ((p)[(offset)] = (data))
#define  vloadN(offset, p) ((p)[(offset)])
#endif

floatN getDisparity(
        __local char *data1,
        __local char *data2,
        int x,
        int y,
        int w,
        int h,
        intN xDelta,
        int minDisparity,
        int maxDisparity,
        int windowSize0,
        int sz,
        floatN* resultVariance,
        bool debug
        BC_ARG
) {
    __local const char *src = data1 + y * sz + x * h * sz;
    __local const char *dest = data2 + y * sz + x * h * sz;

    size_t dataSize = w * sz * h;

    floatN disparity;
    float avgScore;
    maxDisparity = max(minDisparity, maxDisparity);
    minDisparity = min(minDisparity, maxDisparity);
    int disparityRange = maxDisparity - minDisparity;

    CHECK_LOCAL_BOUNDARY(src, data1, dataSize);
    CHECK_LOCAL_BOUNDARY(dest, data2, dataSize);
    CHECK_LOCAL_BOUNDARY(&dest[minDisparity], data2, dataSize);

    intN bestD = (intN)0;

    floatN scoreSum = (floatN)0;
    int currentScoreSize = 1;

    int xDeltaArray[VECTOR_SIZE];
    vstoreN(xDelta, 0, xDeltaArray);

    floatN minCost = (floatN)1e12;

    floatN scoreVariance = (floatN)0;
    floatN scoreExpectation = (floatN)0;

    for (int d = minDisparity; d <= maxDisparity; d++) {
        // предположим что окно соответствия равно 4*4, а maxDisparity-minDisparity кратно 16
        // а формат входных данных: src[tileNo][x1..xn][y1..y4][ch]
        // в score0 поместим стоимость первой колонки пикселей для x
#if VECTOR_SIZE == 1
        const int xDeltaS0 = xDelta;
#else
        const int xDeltaS0 = xDelta.s0;
#endif

        float score0[VECTOR_SIZE + 4];
        float scores[VECTOR_SIZE];
        vstoreN((floatN) 0, 0, scores);
        // в score0 поместим стоимость всего тайла от 0 до N-1 = scoreX0 = score0 + score1 + .. + scoreN
        float score16 = 0;
        // тогда score x1 scoreX1 = score1 + .. + scoreN+1 = score - score0 + scoreN+1

        const char16 d0 = vload16(0, &src[xDeltaS0 * h * sz]) - vload16(0, &dest[d * h * sz + xDeltaS0 * h * sz]);
        const char16 d1 = vload16(1, &src[xDeltaS0 * h * sz]) - vload16(1, &dest[d * h * sz + xDeltaS0 * h * sz]);
        const char16 d2 = vload16(2, &src[xDeltaS0 * h * sz]) - vload16(2, &dest[d * h * sz + xDeltaS0 * h * sz]);
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

        scores[0] = score16;

        char16 c1[VECTOR_SIZE];
        char16 c2[VECTOR_SIZE];
        __attribute__((opencl_unroll_hint(VECTOR_SIZE)))
        for (int xi = 1; xi < VECTOR_SIZE; xi++) {
            c1[xi] = (char16) (
                vload3((xi + 4 - 1 + xDeltaArray[xi]) * h + 1, src),
                vload3((xi + 4 - 1 + xDeltaArray[xi]) * h + 2, src),
                vload3((xi + 4 - 1 + xDeltaArray[xi]) * h + 3, src),
                vload3((xi + 4 - 1 + xDeltaArray[xi]) * h + 4, src),
                0, 0, 0, 0
            );
            c2[xi] = (char16) (
                vload3((xi + 4 - 1 + d) * h + 1, dest),
                vload3((xi + 4 - 1 + d) * h + 2, dest),
                vload3((xi + 4 - 1 + d) * h + 3, dest),
                vload3((xi + 4 - 1 + d) * h + 4, dest),
                0, 0, 0, 0
            );
        }

        __attribute__((opencl_unroll_hint(VECTOR_SIZE)))
        for (int xi = 1; xi < VECTOR_SIZE; xi++) {
            float scoreN = 0;
            float16 cf = convert_float16(c1[xi] - c2[xi]);
            cf *= cf;
            EACH12(scoreN += cf)

            scores[xi] = scores[xi - 1] + scoreN - score0[(xi - 1) % VECTOR_SIZE];
            score0[(xi + 4 - 1) % VECTOR_SIZE] = scoreN;
        }

        floatN newScore = vloadN(0, scores) / (float)windowSize0;
        floatN currentScore = newScore;

        scoreExpectation += newScore;
        scoreVariance += newScore * newScore;

        intN needUpdate = isless(currentScore, minCost);
        needUpdate = needUpdate;
        bestD = select(bestD, (intN)d - xDelta, needUpdate);
        minCost = fmin(minCost, currentScore);
    }

    disparity = convert_floatN(bestD);

    scoreExpectation /= disparityRange;
    scoreVariance /= disparityRange;
    scoreVariance -= scoreExpectation * scoreExpectation;
    scoreVariance = sqrt(fabs(scoreVariance));

    *resultVariance = scoreVariance;

    if (debug) printf("%d %d (%f,%f,%f,%f, %f,%f,%f,%f)\n", minDisparity, x, scoreExpectation.s0, scoreExpectation.s1, scoreExpectation.s2, scoreExpectation.s3, scoreExpectation.s4, scoreExpectation.s5, scoreExpectation.s6, scoreExpectation.s7);

    return fmax(minDisparity, fmin(disparity, maxDisparity));
}

__kernel void DisparityEvaluator(
        __global char* frame0,
        __global char* frame1,
        __global float* roughDisparity,
        __global short* disparity,
        __global float* variance,
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

        for (int x = x0; x < min(xLimit, w - MIN_VALID_DISPARITY - windowSize0); x += VECTOR_SIZE) {
            floatN resultVariance0;
            floatN resultVariance1;
            const int maxDisparity = max(0, min(256, (int) (w - ((x + 1) + windowSize0 + 1))));
            const floatN d0 = getDisparity(pFrame1, pFrame0, x, 0, w, fragmentHeight, (intN)0, 0, maxDisparity, windowSize0, nsz, &resultVariance0, x==960 && y0 == 540 BC_PASS);
            const intN id0 = convert_intN(floor(d0));

            const int minDisparity = max(-x, -256);
            const int maxDisparityX = 0;
#if VECTOR_SIZE == 1
            const int id0s0 = id0;
#else
            const int id0s0 = id0.s0;
#endif
            const floatN xd0 = w - (x + id0s0) < windowSize0 + 1
                    ? d0
                    : -getDisparity(pFrame0, pFrame1, x + id0s0, 0, w, fragmentHeight, id0 - (intN)id0s0, minDisparity, maxDisparityX, windowSize0, nsz, &resultVariance1, false BC_PASS);
            const floatN xd = xd0 * (float) DISPARITY_PRECISION;
            const shortN sxd = convert_shortN(rint(xd));
            const shortN sd = convert_shortN(rint(d0 * (float) DISPARITY_PRECISION));

            vstoreN(resultVariance1, 0, &variance[y * w + x]);
            vstoreN(sxd, 0, &disparity[y * w + x]);
            disparity[y * w + x] = disparity[y * w + x + 1];
        }
    }

    BC_DUMP("(%d;%d)", x0, y0);
//
//    *q += _q - q0;
}
