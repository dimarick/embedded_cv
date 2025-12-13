#ifdef __CLION_IDE__
#include "opencl_ide_defs.h"
#endif

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

#ifndef BATCH_SIZE
#define BATCH_SIZE 1
#endif

#ifndef HALF_FP_AVAILABLE
#define HALF_FP_AVAILABLE 0
#endif

#ifndef MAX_WINDOW_WIDTH
#define MAX_WINDOW_WIDTH 12
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
#define _FATAL_BOUNDARY(type, typeId, _ptr, dataStart, dataSize) \
    if (((type char *)(_ptr) < ((type char *)(dataStart)) || (type char *)(_ptr) >= (((type char *)(dataStart)) + (size_t)(dataSize)))) {\
        __bc->n++; __bc->t = typeId; __bc->ptr = (uintptr_t)(_ptr); __bc->start = (uintptr_t)(dataStart); __bc->end = (uintptr_t)(dataStart) + (size_t)(dataSize); __bc->line = __LINE__;return;}

#define CHECK_GLOBAL_BOUNDARY(_ptr, dataStart, dataSize) _CHECK_BOUNDARY(__global, 'g', _ptr, dataStart, dataSize)
#define CHECK_LOCAL_BOUNDARY(_ptr, dataStart, dataSize) _CHECK_BOUNDARY(__local, 'l', _ptr, dataStart, dataSize)
#define CHECK_BOUNDARY(_ptr, dataStart, dataSize) _CHECK_BOUNDARY(__private, 'p', _ptr, dataStart, dataSize)

#define FATAL_GLOBAL_BOUNDARY(_ptr, dataStart, dataSize) _FATAL_BOUNDARY(__global, 'g', _ptr, dataStart, dataSize)
#define FATAL_LOCAL_BOUNDARY(_ptr, dataStart, dataSize) _FATAL_BOUNDARY(__local, 'l', _ptr, dataStart, dataSize)
#define FATAL_BOUNDARY(_ptr, dataStart, dataSize) _FATAL_BOUNDARY(__private, 'p', _ptr, dataStart, dataSize)

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
#define FATAL_GLOBAL_BOUNDARY(_ptr, dataStart, dataSize)
#define FATAL_LOCAL_BOUNDARY(_ptr, dataStart, dataSize)
#define FATAL_BOUNDARY(_ptr, dataStart, dataSize)
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

#if HALF_FP_AVAILABLE
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#else
#define half16 float16
#define half8 float8
#define half4 float4
#define half2 float2
#define half float

#define convert_half16 convert_float16
#define convert_half8 convert_float8
#define convert_half4 convert_float4
#define convert_half2 convert_float2
#endif

#if VECTOR_SIZE == 16
#define halfN half16
#define intN int16
#define shortN short16
#define charN char16
#define  convert_halfN(...) convert_half16(__VA_ARGS__)
#define  convert_intN(...) convert_int16(__VA_ARGS__)
#define  convert_shortN(...) convert_short16(__VA_ARGS__)
#define  vstoreN(...) vstore16(__VA_ARGS__)
#define  vloadN(...) vload16(__VA_ARGS__)
#elif VECTOR_SIZE == 8
#define halfN half8
#define intN int8
#define shortN short8
#define charN char8
#define  convert_halfN(...) convert_half8(__VA_ARGS__)
#define  convert_intN(...) convert_int8(__VA_ARGS__)
#define  convert_shortN(...) convert_short8(__VA_ARGS__)
#define  vstoreN(...) vstore8(__VA_ARGS__)
#define  vloadN(...) vload8(__VA_ARGS__)
#elif VECTOR_SIZE == 4
#define halfN half4
#define intN int4
#define shortN short4
#define charN char4
#define  convert_halfN(...) convert_half4(__VA_ARGS__)
#define  convert_intN(...) convert_int4(__VA_ARGS__)
#define  convert_shortN(...) convert_short4(__VA_ARGS__)
#define  vstoreN(...) vstore4(__VA_ARGS__)
#define  vloadN(...) vload4(__VA_ARGS__)
#elif VECTOR_SIZE == 2
#define halfN half2
#define intN int2
#define shortN short2
#define charN char2
#define  convert_halfN(...) convert_half2(__VA_ARGS__)
#define  convert_intN(...) convert_int2(__VA_ARGS__)
#define  convert_shortN(...) convert_short2(__VA_ARGS__)
#define  vstoreN(...) vstore2(__VA_ARGS__)
#define  vloadN(...) vload2(__VA_ARGS__)
#elif VECTOR_SIZE == 1
#define halfN half
#define intN int
#define shortN short
#define charN char
#define  convert_halfN(...) (float)(__VA_ARGS__)
#define  convert_intN(...) (int)(__VA_ARGS__)
#define  convert_shortN(...) (short)(__VA_ARGS__)
#define  vstoreN(data, offset, p) ((p)[(offset)] = (data))
#define  vloadN(offset, p) ((p)[(offset)])
#define  select(ifFalse, ifTrue, condition) ((condition) != 0 ? (ifTrue) : (ifFalse))
#endif

void getDisparity(
        __local uchar *data1,
        __local uchar *data2,
        int x,
        int y,
        int w,
        int h,
        short *xDelta,
        int minDisparity,
        int maxDisparity,
        int windowSize0,
        int sz,
        short* result,
        int step,
        bool debug
        BC_ARG
) {
    __local const uchar *src = data1 + y * sz + x * h * sz;
    __local const uchar *dest = data2 + y * sz + x * h * sz;

    size_t dataSize = w * sz * h;

    halfN disparity;
    half avgScore;
    maxDisparity = max(minDisparity, maxDisparity);
    minDisparity = min(minDisparity, maxDisparity);
    int disparityRange = maxDisparity - minDisparity;

    FATAL_LOCAL_BOUNDARY(src, data1, dataSize);
    FATAL_LOCAL_BOUNDARY(dest, data2, dataSize);
    FATAL_LOCAL_BOUNDARY(&dest[minDisparity], data2, dataSize);

    short bestD[VECTOR_SIZE * BATCH_SIZE];
    half minCost[VECTOR_SIZE * BATCH_SIZE];

    for (int b = 0; b < BATCH_SIZE; ++b) {
        vstoreN((shortN)0, b, bestD);
        vstoreN((halfN)1e12f, b, minCost);
    }

    int windowSize = min(MAX_WINDOW_WIDTH, (windowSize0 / 4) * 4);

    for (short d = minDisparity; d <= maxDisparity; d += step) {
        // предположим что окно соответствия равно 4*4, а maxDisparity-minDisparity кратно 16
        // а формат входных данных: src[tileNo][x1..xn][y1..y4][ch]
        // в prev поместим стоимость каждого пикселя первой колонки тайла для текущего d (сумму разностей 16 пикселей для окна 4х4)

        half prev[VECTOR_SIZE + MAX_WINDOW_WIDTH];
        half scores[VECTOR_SIZE * BATCH_SIZE];
        for (int i = 0; i < windowSize; ++i) {
            prev[i] = 0;
        }

        scores[0] = 0;

        FATAL_LOCAL_BOUNDARY(&src[(windowSize / 4 * 3 + 2) * 16], data1, dataSize);
        FATAL_LOCAL_BOUNDARY(&dest[(windowSize / 4 * 3 + 2) * 16 + d * h * sz], data2, dataSize);
        __attribute__((opencl_unroll_hint(2)))
        for (int i = 0; i < windowSize / 4; ++i) {
            half16 df0 = convert_half16(vload16(i * 3 + 0, &src[xDelta[0]])) - convert_half16(vload16(i * 3 + 0, &dest[d * h * sz]));
            half16 df1 = convert_half16(vload16(i * 3 + 1, &src[xDelta[0]])) - convert_half16(vload16(i * 3 + 1, &dest[d * h * sz]));
            half16 df2 = convert_half16(vload16(i * 3 + 2, &src[xDelta[0]])) - convert_half16(vload16(i * 3 + 2, &dest[d * h * sz]));
            df0 *= df0;
            df1 *= df1;
            df2 *= df2;

            prev[i * 4 + 0] += df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4 + df0.s5 + df0.s6 + df0.s7 + df0.s8 + df0.s9 + df0.sA + df0.sB;
            prev[i * 4 + 1] += df0.sC + df0.sD + df0.sE + df0.sF + df1.s0 + df1.s1 + df1.s2 + df1.s3 + df1.s4 + df1.s5 + df1.s6 + df1.s7;
            prev[i * 4 + 2] += df1.s8 + df1.s9 + df1.sA + df1.sB + df1.sC + df1.sD + df1.sE + df1.sF + df2.s0 + df2.s1 + df2.s2 + df2.s3;
            prev[i * 4 + 3] += df2.s4 + df2.s5 + df2.s6 + df2.s7 + df2.s8 + df2.s9 + df2.sA + df2.sB + df2.sC + df2.sD + df2.sE + df2.sF;
            scores[0] += prev[i * 4 + 0] + prev[i * 4 + 1] + prev[i * 4 + 2] + prev[i * 4 + 3];
        }

        // score[0] содержит стоимость disparity d для первого пикселя результата (score1 + .. + scoreN)
        // где score1 - стоимость 1 колонки, scoreN - стоимость N колонки (N = windowSize)

        // тогда score второго пикселя scores[1] = score1 + .. + scoreN+1 = scores[0] - score1 + scoreN+1
        // scores[2] = scores[1] - score2 + scoreN+2
        // где score1 - стоимость 1 колонки, scoreN - стоимость N колонки (N = windowSize)

        // вычисляем score для каждого 1 <= x < BATCH_SIZE * VECTOR_SIZE
        for (int b = 0; b < BATCH_SIZE; ++b) {

            // вычисляем score для каждого b <= x < b + VECTOR_SIZE
            // на первом проходе, где b == 0, пропускаем первую итерацию, так как scores[0] уже вычислен
            for (int xi = (int)(b == 0); xi < VECTOR_SIZE; xi++) {
                int i = b * VECTOR_SIZE + xi;
                int iw = i + windowSize - 1;
                short delta = xDelta[i];
                FATAL_LOCAL_BOUNDARY(&src[(iw + delta) * h * sz], data1, dataSize);
                FATAL_LOCAL_BOUNDARY(&src[(iw + delta) * h * sz + h * sz], data1, dataSize);
                FATAL_LOCAL_BOUNDARY(&dest[(i + d) * h * sz], data2, dataSize);
                FATAL_LOCAL_BOUNDARY(&dest[(i + d) * h * sz + h * sz], data2, dataSize);
                half scoreN = 0;
                for (int j = 0; j < h * sz; ++j) {
                    half c = (half)(src[(iw + delta) * h * sz + j] - dest[(iw + d) * h * sz + j]);
                    scoreN += c * c;
                }
                scores[i] = scores[i - 1] + scoreN - prev[(i - 1) % (VECTOR_SIZE + windowSize)];
                prev[(i + windowSize - 1) % (VECTOR_SIZE + windowSize)] = scoreN;
            }

            halfN currentScore = vloadN(b, scores);

            shortN needUpdate = convert_shortN(isless(currentScore, vloadN(b, minCost)));
            vstoreN(fmin(vloadN(b, minCost), currentScore), b, minCost);
            vstoreN(select(vloadN(b, bestD), (shortN)d - vloadN(b, xDelta), needUpdate), b, bestD);
        }
    }

    for (int b = 0; b < BATCH_SIZE; ++b) {
        vstoreN(vloadN(b, bestD), b, result);
    }
}

__kernel void DisparityEvaluator(
        __global uchar* frame0,
        __global uchar* frame1,
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

    int x0 = (get_local_id(0) / 2) * H_GRANULE_SIZE;
    int y0 = get_global_id(1) * V_GRANULE_SIZE;
    y0 += get_local_id(0) & 1;

    int wsz = w * sz;

    int windowSize0 = 4;

    __local uchar pFrame0[maxLocalBuffer] __attribute__ ((aligned (128)));
    __local uchar pFrame1[maxLocalBuffer] __attribute__ ((aligned (128)));

    int xLimit = min(x0 + H_GRANULE_SIZE, w - 16*3);
    int yLimit = min(y0 + V_GRANULE_SIZE / 2, h - V_GRANULE_SIZE / 2 - 8);

    int fragmentHeight = min(maxFragmentHeight, windowHeight);
    fragmentHeight = min(fragmentHeight, h - 1 - y0);

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

        for (int x = x0; x < min(xLimit, w - MIN_VALID_DISPARITY - windowSize0); x += BATCH_SIZE * VECTOR_SIZE) {
            short result[BATCH_SIZE * VECTOR_SIZE];
//            short result2[BATCH_SIZE * VECTOR_SIZE];
            short xDeltaArray[BATCH_SIZE * VECTOR_SIZE];

            for (short i = 0; i < BATCH_SIZE * VECTOR_SIZE; ++i) {
                xDeltaArray[i] = 0;
            }
            const int maxDisparity = max(0, min(256, (int) (w - ((x + 16*3) + windowSize0 + 1))));
            getDisparity(pFrame1, pFrame0, x, 0, w, fragmentHeight, xDeltaArray,
                         0, maxDisparity, windowSize0, nsz, result, 1, false BC_PASS);

            const short r0 = result[0];
            for (int i = 0; i < BATCH_SIZE * VECTOR_SIZE; ++i) {
                short d = result[i] - r0;
                xDeltaArray[i] = d;
            }

            const int minDisparity = max(-r0 - 64, -256);
            const int maxDisparityX = min(-r0 + 64, 0);

            getDisparity(pFrame0, pFrame1, x + r0, 0, w, fragmentHeight, xDeltaArray,
                                  max(-x - r0, minDisparity), maxDisparityX, windowSize0, nsz, result, 1, false BC_PASS);

//            const short r01 = result[0];
//            for (int i = 0; i < BATCH_SIZE * VECTOR_SIZE; ++i) {
//                short d = result[i] - r01;
//                xDeltaArray[i] = d;
//            }
//
//            const int minDisparity1 = -min(x + r01, 5);
//            int maxDisparityX1 = 5;
//
//            maxDisparityX1 = max(0, min(maxDisparityX1, (int) (w - ((x + 16*3) + windowSize0 + 1))));
//
//            getDisparity(pFrame1, pFrame0, x + r01, 0, w, fragmentHeight, xDeltaArray, minDisparity1,
//                                  maxDisparityX1, windowSize0, nsz, result2, 1, false BC_PASS);

            for (int i = 0; i < BATCH_SIZE * VECTOR_SIZE; ++i) {
                short d = -(result[i]) * DISPARITY_PRECISION;
//                disparity[y * w + x + i] = d * (short)(d < 255 * DISPARITY_PRECISION) + disparity[y * w + x + i - 1] * !(short)(d < 255 * DISPARITY_PRECISION);
                disparity[y * w + x + i] = d;
            }
        }
    }

    BC_DUMP("(%d;%d)", x0, y0);
//
//    *q += _q - q0;
}
