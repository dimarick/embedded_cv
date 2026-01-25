#ifdef __CLION_IDE__
#include "opencl_ide_defs.h"
#endif

#ifndef H_GRANULE_SIZE
#define H_GRANULE_SIZE 16
#endif

#ifndef V_GRANULE_SIZE
#define V_GRANULE_SIZE 2
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 1
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE H_GRANULE_SIZE
#endif

#ifndef HALF_FP_AVAILABLE
#define HALF_FP_AVAILABLE 0
#endif

#ifndef WINDOW_SIZE
#define WINDOW_SIZE 9
#endif

#ifndef MAX_DISPARITY
#define MAX_DISPARITY 384
#endif

#ifndef DISPARITY_PRECISION
#define DISPARITY_PRECISION 16
#endif

#ifndef N_CANDIDATES
#define N_CANDIDATES 2
#endif

#ifndef EPS
#define EPS 1e-3
#endif

#define MAX_FRAGMENT_HEIGHT WINDOW_SIZE
#define N_CHANNELS 3
#define MAX_ROW_WIDTH 1920
#define MAX_LOCAL_BUFFER MAX_ROW_WIDTH * (MAX_FRAGMENT_HEIGHT)
#define MIN_VALID_DISPARITY 5

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
#define half3 float3
#define half2 float2
#define half float

#define convert_half16 convert_float16
#define convert_half8 convert_float8
#define convert_half4 convert_float4
#define convert_half3 convert_float3
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

void inline appendScore9x9(const __local uchar *src, const __local uchar *dest, half *perColumnScore) {
    half16 df0 = convert_half16(vload16(0, src)) - convert_half16(vload16(0, dest));
    half16 df1 = convert_half16(vload16(1, src)) - convert_half16(vload16(1, dest));
    half16 df2 = convert_half16(vload16(2, src)) - convert_half16(vload16(2, dest));
    half16 df3 = convert_half16(vload16(3, src)) - convert_half16(vload16(3, dest));
    half16 df4 = convert_half16(vload16(4, src)) - convert_half16(vload16(4, dest));
    half tail = (half)src[9*9-1] - (half)dest[9*9-1];
    df0 *= df0;
    df1 *= df1;
    df2 *= df2;
    df3 *= df3;
    df4 *= df4;
    tail *= tail;

    perColumnScore[0] += df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4 + df0.s5 + df0.s6 + df0.s7 + df0.s8;
    perColumnScore[1] += df0.s9 + df0.sA + df0.sB + df0.sC + df0.sD + df0.sE + df0.sF + df1.s0 + df1.s1;
    perColumnScore[2] += df1.s2 + df1.s3 + df1.s4 + df1.s5 + df1.s6 + df1.s7 + df1.s8 + df1.s9 + df1.sA;
    perColumnScore[3] += df1.sB + df1.sC + df1.sD + df1.sE + df1.sF + df2.s0 + df2.s1 + df2.s2 + df2.s3;
    perColumnScore[4] += df2.s4 + df2.s5 + df2.s6 + df2.s7 + df2.s8 + df2.s9 + df2.sA + df2.sB + df2.sC;
    perColumnScore[5] += df2.sD + df2.sE + df2.sF + df3.s0 + df3.s1 + df3.s2 + df3.s3 + df3.s4 + df3.s5;
    perColumnScore[6] += df3.s6 + df3.s7 + df3.s8 + df3.s9 + df3.sA + df3.sB + df3.sC + df3.sD + df3.sE;
    perColumnScore[7] += df3.sF + df3.s0 + df4.s1 + df4.s2 + df4.s3 + df4.s4 + df4.s5 + df4.s6 + df4.s7;
    perColumnScore[8] += df4.s8 + df4.s9 + df4.sA + df4.sB + df4.sC + df4.sD + df4.sE + df4.sF + tail;
}

half inline appendScore9x1(const __local uchar *src, const __local uchar *dest) {
    half8 df0 = convert_half8(vload8(0, src)) - convert_half8(vload8(0, dest));
    half tail = (half)src[9-1] - (half)dest[9-1];
    df0 *= df0;
    tail *= tail;

    return df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4 + df0.s5 + df0.s6 + df0.s7 + tail;
}

void inline appendScore7x7(const __local uchar *src, const __local uchar *dest, half *perColumnScore) {
    half16 df0 = convert_half16(vload16(0, src)) - convert_half16(vload16(0, dest));
    half16 df1 = convert_half16(vload16(1, src)) - convert_half16(vload16(1, dest));
    half16 df2 = convert_half16(vload16(2, src)) - convert_half16(vload16(2, dest));
    half tail = (half)src[7*7-1] - (half)dest[7*7-1];
    df0 *= df0;
    df1 *= df1;
    df2 *= df2;
    tail *= tail;

    perColumnScore[0] += df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4 + df0.s5 + df0.s6;
    perColumnScore[1] += df0.s7 + df0.s8 + df0.s9 + df0.sA + df0.sB + df0.sC + df0.sD;
    perColumnScore[2] += df0.sE + df0.sF + df1.s0 + df1.s1 + df1.s2 + df1.s3 + df1.s4;
    perColumnScore[3] += df1.s5 + df1.s6 + df1.s7 + df1.s8 + df1.s9 + df1.sA + df1.sB;
    perColumnScore[4] += df1.sC + df1.sD + df1.sE + df1.sF + df2.s0 + df2.s1 + df2.s2;
    perColumnScore[5] += df2.s3 + df2.s4 + df2.s5 + df2.s6 + df2.s7 + df2.s8 + df2.s9;
    perColumnScore[6] += df2.sA + df2.sB + df2.sC + df2.sD + df2.sE + df2.sF + tail;
}

half inline appendScore7x1(const __local uchar *src, const __local uchar *dest) {
    half8 df0 = convert_half8(vload8(0, src)) - convert_half8(vload8(0, dest));

    return df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4 + df0.s5 + df0.s6;
}

void inline appendScore5x5(const __local uchar *src, const __local uchar *dest, half *perColumnScore) {
    half16 df0 = convert_half16(vload16(0, src)) - convert_half16(vload16(0, dest));
    half8 df1 = convert_half8(vload8(1, src)) - convert_half8(vload8(1, dest));
    half tail = (half)src[5*5-1] - (half)dest[5*5-1];
    df0 *= df0;
    df1 *= df1;
    tail *= tail;

    perColumnScore[0] += df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4;
    perColumnScore[1] += df0.s5 + df0.s6 + df0.s7 + df0.s8 + df0.s9;
    perColumnScore[2] += df0.sA + df0.sB + df0.sC + df0.sD + df0.sE;
    perColumnScore[3] += df0.sF + df1.s0 + df1.s1 + df1.s2 + df1.s3;
    perColumnScore[4] += df1.s4 + df1.s5 + df1.s6 + df1.s7 + tail;
}

half inline appendScore5x1(const __local uchar *src, const __local uchar *dest) {
    half4 df0 = convert_half4(vload4(0, src)) - convert_half4(vload4(0, dest));
    half tail = (half)src[5-1] - (half)dest[5-1];
    df0 *= df0;
    tail *= tail;

    return df0.s0 + df0.s1 + df0.s2 + df0.s3 + tail;
}

void inline appendScore3x3(const __local uchar *src, const __local uchar *dest, half *perColumnScore) {
    half8 df0 = convert_half8(vload8(1, src)) - convert_half8(vload8(1, dest));
    half tail = (half)src[3*3-1] - (half)dest[3*3-1];
    df0 *= df0;
    tail *= tail;

    perColumnScore[0] += df0.s0 + df0.s1 + df0.s2;
    perColumnScore[1] += df0.s3 + df0.s4 + df0.s5;
    perColumnScore[2] += df0.s6 + df0.s7 + tail;
}

half inline appendScore3x1(const __local uchar *src, const __local uchar *dest) {
    half3 df0 = convert_half3(vload3(0, src)) - convert_half3(vload3(0, dest));
    df0 *= df0;

    return df0.s0 + df0.s1 + df0.s2;
}

half inline sum(half16 v16) {
    half8 v8 = v16.odd + v16.even;
    half4 v4 = v8.odd + v8.even;
    half2 v2 = v4.odd + v4.even;

    return v2.odd + v2.even;
}

half inline mean(half16 v) {
    return sum(v) / 16;
}

half inline zstddev(half16 v) {
    v *= v;
    half r = sum(v);

    return sqrt(r) + EPS;
}

half inline getWindowColumnsScore(const __local uchar *src, const __local uchar *dest, half *perColumnScore) {
    for (int i = 0; i < WINDOW_SIZE; ++i) {
        perColumnScore[i] = 0;
    }

    switch (WINDOW_SIZE) {
        case 3:
            appendScore3x3(src, dest, perColumnScore);
        case 5:
            appendScore5x5(src, dest, perColumnScore);
        case 7:
            appendScore7x7(src, dest, perColumnScore);
        case 9:
            appendScore9x9(src, dest, perColumnScore);
    }

    half scores = 0;
    for (int i = 0; i < WINDOW_SIZE; ++i) {
        scores += perColumnScore[i];
    }

    return scores;
}

half inline getWindowColumnScore(const __local uchar *src, const __local uchar *dest) {
    switch (WINDOW_SIZE) {
        case 3:
            return appendScore3x1(src, dest);
        case 5:
            return appendScore5x1(src, dest);
        case 7:
            return appendScore7x1(src, dest);
        case 9:
            return appendScore9x1(src, dest);
    }
}

void getDisparityCandidates(
        __local uchar *data1,
        __local uchar *data2,
        int x,
        int y,
        int w,
        int h,
        int minDisparity,
        int maxDisparity,
        int windowSize,
        int sz,
        short* result,
        half* costs,
        int nCandidates,
        int batchSize,
        int step,
        bool debug
        BC_ARG
) {
    __local const uchar *src = data1 + y * sz + x * h * sz;
    __local const uchar *dest = data2 + y * sz + x * h * sz;

    size_t dataSize = w * sz * h;

    int disparityRange = maxDisparity - minDisparity;

    FATAL_LOCAL_BOUNDARY(src, data1, dataSize);
    FATAL_LOCAL_BOUNDARY(dest, data2, dataSize);
    FATAL_LOCAL_BOUNDARY(&dest[minDisparity], data2, dataSize);

    short stepScale = 1;

    for (short d = minDisparity; d <= maxDisparity; d += step * stepScale) {
        half perColumnScore[BATCH_SIZE + WINDOW_SIZE + 1];

        half cost = getWindowColumnsScore(src, &dest[d * h * sz], &perColumnScore[0]);

        for (int b = 0; b < batchSize; ++b) {
            int bw = b + WINDOW_SIZE;
            perColumnScore[bw] = getWindowColumnScore(src + bw * h * sz, &dest[d * h * sz + bw * h * sz]);

            if (abs(result[b] - d) > step * stepScale && cost < costs[b]) {
                for (int k = nCandidates - 2; k >= 0; k--) {
                    result[(k + 1) * BATCH_SIZE + b] = result[k * BATCH_SIZE + b];
                    costs[(k + 1) * BATCH_SIZE + b] = costs[k * BATCH_SIZE + b];
                }
            }

            bool needUpdate = cost < costs[b];
            costs[b] = fmin(costs[b], cost);
            result[b] = needUpdate ? d : result[b];

            cost = cost - perColumnScore[b] + perColumnScore[bw];
        }
    }
}

__kernel void DisparityEvaluator(
        __global uchar* frame0,
        __global uchar* frame1,
        __global short* disparity,
//        __global float* variance,
        const int w,
        const int h,
        const int channels
) {
    BC_START;

    int x0 = (get_local_id(0) / 2) * H_GRANULE_SIZE;
    int y0 = get_global_id(1) * V_GRANULE_SIZE;
    y0 += get_local_id(0) & 1;

    int wchannels = w * channels;

    int windowSize0 = 7;
    windowSize0 = min(WINDOW_SIZE, (windowSize0 / 7) * 7);

    __local uchar pFrame0[MAX_LOCAL_BUFFER] __attribute__ ((aligned (128)));
    __local uchar pFrame1[MAX_LOCAL_BUFFER] __attribute__ ((aligned (128)));

    int xLimit = min(x0 + H_GRANULE_SIZE, w - WINDOW_SIZE);
    int yLimit = min(y0 + V_GRANULE_SIZE / 2, h - V_GRANULE_SIZE / 2 - MAX_FRAGMENT_HEIGHT);

    int fragmentHeight = MAX_FRAGMENT_HEIGHT;
    fragmentHeight = min(fragmentHeight, h - 1 - y0);

    for (int y = y0; y < yLimit; ++y) {
        if (channels == 3) {
            half chWeight[3] = {0.299, 0.587, 0.114};

            for (int r = 0; r < MAX_FRAGMENT_HEIGHT; ++r) {
                for (int x = x0; x < xLimit; x++) {
                    half c0 = 0;
                    half c1 = 0;
                    for (int ch = 0; ch < channels; ch++) {
                        c0 += chWeight[ch] * (half) frame0[(y + r) * wchannels + x * channels + ch];
                        c1 += chWeight[ch] * (half) frame1[(y + r) * wchannels + x * channels + ch];
                    }
                    pFrame0[r + x * MAX_FRAGMENT_HEIGHT] = c0;
                    pFrame1[r + x * MAX_FRAGMENT_HEIGHT] = c1;
                }
            }
        } else {
            for (int r = 0; r < MAX_FRAGMENT_HEIGHT; ++r) {
                for (int x = x0; x < xLimit; x++) {
                    pFrame0[r + x * MAX_FRAGMENT_HEIGHT] = frame0[(y + r) * w + x];
                    pFrame1[r + x * MAX_FRAGMENT_HEIGHT] = frame1[(y + r) * w + x];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        short disparities[BATCH_SIZE * N_CANDIDATES];
        half costs[BATCH_SIZE * N_CANDIDATES];

        for (int x = x0; x < min(xLimit, w - MIN_VALID_DISPARITY - windowSize0); x += BATCH_SIZE) {
            for (int i = 0; i < BATCH_SIZE; ++i) {
                disparities[i] = 0;
                costs[i] = 1e12;
            }

            for (int i = BATCH_SIZE; i < BATCH_SIZE * N_CANDIDATES; ++i) {
                disparities[i] = 0;
                costs[i] = 0;
            }

            const int maxDisparity = max(0, min(MAX_DISPARITY, (int) (w - (x + windowSize0 + 1))));

            getDisparityCandidates(pFrame0, pFrame1, x, 0, w, MAX_FRAGMENT_HEIGHT,
                         max(-x, -MAX_DISPARITY), 0, WINDOW_SIZE, 1, disparities, costs, N_CANDIDATES, BATCH_SIZE, 1, false BC_PASS);

//
//            for (int i = 0; i < BATCH_SIZE; ++i) {
//                short d = disparities[i];
//                getDisparityCandidates(pFrame0, pFrame1, x, 0, w, MAX_FRAGMENT_HEIGHT,
//                                       max(-x, d - 2), min(maxDisparity, d + 2), WINDOW_SIZE, 1, &disparities[i], &costs[i], N_CANDIDATES, 1, 1, false BC_PASS);
//            }


            short d0 = abs(disparities[0]);

//            getDisparityCandidates(pFrame1, pFrame0, x + d, 0, w, MAX_FRAGMENT_HEIGHT,
//                                   max(-x, -d - 4), min(maxDisparity, -d + 4), WINDOW_SIZE, 1, disparities, costs, 1, 1, false BC_PASS);

            for (int b = 0; b < BATCH_SIZE; ++b) {
                short c0 =  abs((short)(costs[b]));
                short c1 = abs((short)(costs[b + BATCH_SIZE]));
                short d = abs((short)disparities[b]);
                bool valid = true;
//                disparity[(y + WINDOW_SIZE / 2) * w + x + WINDOW_SIZE / 2 + b] = d * DISPARITY_PRECISION;
                disparity[(y + WINDOW_SIZE / 2) * w + x + WINDOW_SIZE / 2 + b] = valid
                     && c1 / (c0 + EPS) > 1.15
                     && d < (MAX_DISPARITY - 64) ?
                     d * DISPARITY_PRECISION :
                     0;
            }
        }
    }

    BC_DUMP("(%d;%d)", x0, y0);
}
