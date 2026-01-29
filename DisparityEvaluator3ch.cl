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

#ifndef WINDOW_SIZE
#define WINDOW_SIZE 11
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif

#ifndef HALF_FP_AVAILABLE
#define HALF_FP_AVAILABLE 0
#endif

#ifndef MAX_DISPARITY
#define MAX_DISPARITY 384
#endif

#ifndef DISPARITY_SCALE
#define DISPARITY_SCALE 16
#endif

#ifndef N_CANDIDATES
#define N_CANDIDATES 2
#endif

#ifndef EPS
#define EPS 1e-6
#endif

#define MAX_FRAGMENT_HEIGHT WINDOW_SIZE
#define N_CHANNELS 3
#define MAX_ROW_WIDTH 1920
#define MAX_LOCAL_BUFFER MAX_ROW_WIDTH * (MAX_FRAGMENT_HEIGHT)
#define MIN_VALID_DISPARITY 5

#define SCORE_NORM (1.0 / (256 * WINDOW_SIZE * WINDOW_SIZE))

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

half inline sum16(half16 v16) {
    half8 v8 = v16.odd + v16.even;
    half4 v4 = v8.odd + v8.even;
    half2 v2 = v4.odd + v4.even;

    return v2.odd + v2.even;
}

half inline sum8(half8 v8) {
    half4 v4 = v8.odd + v8.even;
    half2 v2 = v4.odd + v4.even;

    return v2.odd + v2.even;
}

half inline sum4(half4 v4) {
    half2 v2 = v4.odd + v4.even;

    return v2.odd + v2.even;
}

half inline sum3(half3 v3) {
    return v3.s0 + v3.s1 + v3.s2;
}

short inline sums8(short8 v8) {
    short4 v4 = v8.odd + v8.even;
    short2 v2 = v4.odd + v4.even;

    return v2.odd + v2.even;
}

half inline mean(half16 v) {
    return sum16(v) / 16;
}

half inline zstddev(half16 v) {
    v *= v;
    half r = sum16(v);

    return sqrt(r) + EPS;
}

void inline loadPatch11x1(const uchar *src, half *patch) {
    vstore8(convert_half8(vload8(0, src)), 0, patch);
    vstore3(convert_half3(vload3(0, src + 8)), 0, patch + 8);
}

void inline loadPatch11x11(const uchar *src, half *patch) {
    vstore16(convert_half16(vload16(0, src)), 0, patch);
    vstore16(convert_half16(vload16(1, src)), 1, patch);
    vstore16(convert_half16(vload16(2, src)), 2, patch);
    vstore16(convert_half16(vload16(3, src)), 3, patch);
    vstore16(convert_half16(vload16(4, src)), 4, patch);
    vstore16(convert_half16(vload16(5, src)), 5, patch);
    vstore16(convert_half16(vload16(6, src)), 6, patch);
    vstore8(convert_half8(vload8(0, src + 16 * 7)), 0, patch + 16 * 7);
    patch[11*11-1] = (half)src[11*11-1];
}

half inline getPatchScore11x11(half *patch) {
    half score = 0;

    for (int i = 0; i < 10; ++i) {
        half8 df0 = vload8(0, patch + 11 * i) - vload8(0, patch + 11 * i + 11);
        half3 df1 = vload3(0, patch + 11 * i) - vload3(0, patch + 11 * i + 11 + 8);
        df0 *= df0;
        df1 *= df1;

        score += sum8(df0) + sum3(df1);
    }

    return score / (10 * 11 * 256);
}

half inline getPatchScore9x9(half *patch) {
    half score = 0;

    for (int i = 0; i < 8; ++i) {
        half8 df0 = vload8(0, patch + 9 * i) - vload8(0, patch + 9 * i + 9);
        half tail = patch[9-1 + 9 * i] - (half)patch[9-1 + 9 * i + 9];
        df0 *= df0;
        tail *= tail;

        score += sum8(df0) + tail;
    }

    return score / (8 * 9 * 256);
}

half inline getPatchScore7x7(half *patch) {
    half score = 0;

    for (int i = 0; i < 6; ++i) {
        half4 df0 = vload4(0, patch + 7 * i) - vload4(0, patch + 7 * i + 7);
        half3 df1 = vload3(0, patch + 7 * i) - vload3(0, patch + 7 * i + 7 + 4);
        df0 *= df0;
        df1 *= df1;

        score += sum4(df0) + sum3(df1);
    }

    return score / (6 * 7 * 256);
}

half inline getPatchScore5x5(half *patch) {
    half score = 0;

    for (int i = 0; i < 4; ++i) {
        half4 df0 = vload4(0, patch + 5 * i) - vload4(0, patch + 5 * i + 5);
        half tail = patch[5-1 + 5 * i] - (half)patch[5-1 + 5 * i + 5];
        df0 *= df0;
        tail *= tail;

        score += sum4(df0) + tail;
    }

    return score / (4 * 5 * 256);
}

half inline getPatchScore3x3(half *patch) {
    half score = 0;

    for (int i = 0; i < 2; ++i) {
        half3 df0 = vload3(0, patch + 3 * i) - vload3(0, patch + 3 * i + 3);
        df0 *= df0;

        score += sum3(df0);
    }

    return score / (2 * 3 * 256);
}

void inline appendScore11x11(const half *src, const __local uchar *dest, half *perColumnScore) {
    half16 df0 = vload16(0, src) - convert_half16(vload16(0, dest));
    half16 df1 = vload16(1, src) - convert_half16(vload16(1, dest));
    half16 df2 = vload16(2, src) - convert_half16(vload16(2, dest));
    half16 df3 = vload16(3, src) - convert_half16(vload16(3, dest));
    half16 df4 = vload16(4, src) - convert_half16(vload16(4, dest));
    half16 df5 = vload16(5, src) - convert_half16(vload16(5, dest));
    half16 df6 = vload16(6, src) - convert_half16(vload16(6, dest));
    half8 df7 = vload8(0, src + 16 * 7) - convert_half8(vload8(0, dest + 16 * 7));
    half tail = src[11*11-1] - (half)(dest[11*11-1]);
    df0 *= df0 * (half16)SCORE_NORM;
    df1 *= df1 * (half16)SCORE_NORM;
    df2 *= df2 * (half16)SCORE_NORM;
    df3 *= df3 * (half16)SCORE_NORM;
    df4 *= df4 * (half16)SCORE_NORM;
    df5 *= df5 * (half16)SCORE_NORM;
    df6 *= df6 * (half16)SCORE_NORM;
    df7 *= df7 * (half8)SCORE_NORM;
    tail *= tail * SCORE_NORM;

    perColumnScore[0] += df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4 + df0.s5 + df0.s6 + df0.s7 + df0.s8 + df0.s9 + df0.sA;
    perColumnScore[1] += df0.sB + df0.sC + df0.sD + df0.sE + df0.sF + df1.s0 + df1.s1 + df1.s2 + df1.s3 + df1.s4 + df1.s5;
    perColumnScore[2] += df1.s6 + df1.s7 + df1.s8 + df1.s9 + df1.sA + df1.sB + df1.sC + df1.sD + df1.sE + df1.sF + df2.s0;
    perColumnScore[3] += df2.s1 + df2.s2 + df2.s3 + df2.s4 + df2.s5 + df2.s6 + df2.s7 + df2.s8 + df2.s9 + df2.sA + df2.sB;
    perColumnScore[4] += df2.sC + df2.sD + df2.sE + df2.sF + df3.s0 + df3.s1 + df3.s2 + df3.s3 + df3.s4 + df3.s5 + df3.s6;
    perColumnScore[5] += df3.s7 + df3.s8 + df3.s9 + df3.sA + df3.sB + df3.sC + df3.sD + df3.sE + df3.sF + df4.s0 + df4.s1;
    perColumnScore[6] += df4.s2 + df4.s3 + df4.s4 + df4.s5 + df4.s6 + df4.s7 + df4.s8 + df4.s9 + df4.sA + df4.sB + df4.sC;
    perColumnScore[7] += df4.sD + df4.sE + df4.sF + df5.s0 + df5.s1 + df5.s2 + df5.s3 + df5.s5 + df5.s5 + df5.s6 + df5.s7;
    perColumnScore[8] += df5.s8 + df5.s9 + df5.sA + df5.sB + df5.sC + df5.sD + df5.sE + df5.sF + df6.s0 + df6.s1 + df6.s2;
    perColumnScore[9] += df6.s3 + df6.s6 + df6.s6 + df6.s6 + df6.s7 + df6.s8 + df6.s9 + df6.sA + df6.sB + df6.sC + df6.sD;
    perColumnScore[10]+= df6.sE + df6.sF + df7.s0 + df7.s1 + df7.s2 + df7.s3 + df7.s7 + df7.s7 + df7.s6 + df7.s7 + tail;
}

half inline appendScore11x1(const half *src, const __local uchar *dest) {
    half8 df0 = vload8(0, src) - convert_half8(vload8(0, dest));
    half3 df1 = vload3(0, src + 8) - convert_half3(vload3(0, dest + 8));
    df0 *= df0 * (half8)SCORE_NORM;
    df1 *= df1 * (half3)SCORE_NORM;

    return sum8(df0) + sum3(df1);
}

void inline loadPatch9x1(const uchar *src, half *patch) {
    vstore8(convert_half8(vload8(0, src)), 0, patch);
    patch[9-1] = (half)src[9-1];
}

void inline loadPatch9x9(const uchar *src, half *patch) {
    vstore16(convert_half16(vload16(0, src)), 0, patch);
    vstore16(convert_half16(vload16(1, src)), 1, patch);
    vstore16(convert_half16(vload16(2, src)), 2, patch);
    vstore16(convert_half16(vload16(3, src)), 3, patch);
    vstore16(convert_half16(vload16(4, src)), 4, patch);
    patch[9*9-1] = (half)src[9*9-1];
}

void inline appendScore9x9(const half *src, const __local uchar *dest, half *perColumnScore) {
    half16 df0 = vload16(0, src) - convert_half16(vload16(0, dest));
    half16 df1 = vload16(1, src) - convert_half16(vload16(1, dest));
    half16 df2 = vload16(2, src) - convert_half16(vload16(2, dest));
    half16 df3 = vload16(3, src) - convert_half16(vload16(3, dest));
    half16 df4 = vload16(4, src) - convert_half16(vload16(4, dest));
    half tail = src[9*9-1] - (half)(dest[9*9-1]);
    df0 *= df0 * (half16)SCORE_NORM;
    df1 *= df1 * (half16)SCORE_NORM;
    df2 *= df2 * (half16)SCORE_NORM;
    df3 *= df3 * (half16)SCORE_NORM;
    df4 *= df4 * (half16)SCORE_NORM;
    tail *= tail * SCORE_NORM;

    perColumnScore[0] += df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4 + df0.s5 + df0.s6 + df0.s7 + df0.s8;
    perColumnScore[1] += df0.s9 + df0.sA + df0.sB + df0.sC + df0.sD + df0.sE + df0.sF + df1.s0 + df1.s1;
    perColumnScore[2] += df1.s2 + df1.s3 + df1.s4 + df1.s5 + df1.s6 + df1.s7 + df1.s8 + df1.s9 + df1.sA;
    perColumnScore[3] += df1.sB + df1.sC + df1.sD + df1.sE + df1.sF + df2.s0 + df2.s1 + df2.s2 + df2.s3;
    perColumnScore[4] += df2.s4 + df2.s5 + df2.s6 + df2.s7 + df2.s8 + df2.s9 + df2.sA + df2.sB + df2.sC;
    perColumnScore[5] += df2.sD + df2.sE + df2.sF + df3.s0 + df3.s1 + df3.s2 + df3.s3 + df3.s4 + df3.s5;
    perColumnScore[6] += df3.s6 + df3.s7 + df3.s8 + df3.s9 + df3.sA + df3.sB + df3.sC + df3.sD + df3.sE;
    perColumnScore[7] += df3.sF + df4.s0 + df4.s1 + df4.s2 + df4.s3 + df4.s4 + df4.s5 + df4.s6 + df4.s7;
    perColumnScore[8] += df4.s8 + df4.s9 + df4.sA + df4.sB + df4.sC + df4.sD + df4.sE + df4.sF + tail;
}

half inline appendScore9x1(const half *src, const __local uchar *dest) {
    half8 df0 = vload8(0, src) - convert_half8(vload8(0, dest));
    half tail = src[9-1] - (half)dest[9-1];
    df0 *= df0 * (half8)SCORE_NORM;
    tail *= tail * SCORE_NORM;

    return sum8(df0) + tail;
}

void inline loadPatch7x1(const uchar *src, half *patch) {
    vstore4(convert_half4(vload4(0, src)), 0, patch);
    vstore3(convert_half3(vload3(0, src + 4)), 0, patch + 4);
}

void inline loadPatch7x7(const uchar *src, half *patch) {
    vstore16(convert_half16(vload16(0, src)), 0, patch);
    vstore16(convert_half16(vload16(1, src)), 1, patch);
    vstore16(convert_half16(vload16(2, src)), 2, patch);
    patch[7*7-1] = (half)src[7*7-1];
}

void inline appendScore7x7(const half *src, const __local uchar *dest, half *perColumnScore) {
    half16 df0 = vload16(0, src) - convert_half16(vload16(0, dest));
    half16 df1 = vload16(1, src) - convert_half16(vload16(1, dest));
    half16 df2 = vload16(2, src) - convert_half16(vload16(2, dest));
    half tail = src[7*7-1] - (half)dest[7*7-1];
    df0 *= df0 * (half16)SCORE_NORM;
    df1 *= df1 * (half16)SCORE_NORM;
    df2 *= df2 * (half16)SCORE_NORM;
    tail *= tail * SCORE_NORM;

    perColumnScore[0] += df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4 + df0.s5 + df0.s6;
    perColumnScore[1] += df0.s7 + df0.s8 + df0.s9 + df0.sA + df0.sB + df0.sC + df0.sD;
    perColumnScore[2] += df0.sE + df0.sF + df1.s0 + df1.s1 + df1.s2 + df1.s3 + df1.s4;
    perColumnScore[3] += df1.s5 + df1.s6 + df1.s7 + df1.s8 + df1.s9 + df1.sA + df1.sB;
    perColumnScore[4] += df1.sC + df1.sD + df1.sE + df1.sF + df2.s0 + df2.s1 + df2.s2;
    perColumnScore[5] += df2.s3 + df2.s4 + df2.s5 + df2.s6 + df2.s7 + df2.s8 + df2.s9;
    perColumnScore[6] += df2.sA + df2.sB + df2.sC + df2.sD + df2.sE + df2.sF + tail;
}

half inline appendScore7x1(const half *src, const __local uchar *dest) {
    half8 df0 = vload8(0, src) - convert_half8(vload8(0, dest));
    df0 *= df0 * (half8)SCORE_NORM;

    return df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4 + df0.s5 + df0.s6;
}

void inline loadPatch5x1(const uchar *src, half *patch) {
    vstore3(convert_half3(vload3(0, src)), 0, patch);
    vstore2(convert_half2(vload2(0, src + 3)), 0, patch + 3);
}

void inline loadPatch5x5(const uchar *src, half *patch) {
    vstore16(convert_half16(vload16(0, src)), 0, patch);
    vstore8(convert_half8(vload8(0, src + 16)), 0, patch + 16);
    patch[5*5-1] = (half)src[5*5-1];
}

void inline appendScore5x5(const half *src, const __local uchar *dest, half *perColumnScore) {
    half16 df0 = vload16(0, src) - convert_half16(vload16(0, dest));
    half8 df1 = vload8(1, src) - convert_half8(vload8(1, dest));
    half tail = src[5*5-1] - (half)dest[5*5-1];
    df0 *= df0 * (half16)SCORE_NORM;
    df1 *= df1 * (half8)SCORE_NORM;
    tail *= tail * SCORE_NORM;

    perColumnScore[0] += df0.s0 + df0.s1 + df0.s2 + df0.s3 + df0.s4;
    perColumnScore[1] += df0.s5 + df0.s6 + df0.s7 + df0.s8 + df0.s9;
    perColumnScore[2] += df0.sA + df0.sB + df0.sC + df0.sD + df0.sE;
    perColumnScore[3] += df0.sF + df1.s0 + df1.s1 + df1.s2 + df1.s3;
    perColumnScore[4] += df1.s4 + df1.s5 + df1.s6 + df1.s7 + tail;
}

half inline appendScore5x1(const half *src, const __local uchar *dest) {
    half4 df0 = vload4(0, src) - convert_half4(vload4(0, dest));
    half tail = src[5-1] - (half)dest[5-1];
    df0 *= df0 * (half4)SCORE_NORM;
    tail *= tail * SCORE_NORM;

    return df0.s0 + df0.s1 + df0.s2 + df0.s3 + tail;
}

void inline loadPatch3x1(const uchar *src, half *patch) {
    vstore3(convert_half3(vload3(0, src)), 0, patch);
}

void inline loadPatch3x3(const uchar *src, half *patch) {
    vstore8(convert_half8(vload8(0, src)), 0, patch);
    patch[3*3-1] = (half)src[3*3-1];
}

void inline appendScore3x3(const half *src, const __local uchar *dest, half *perColumnScore) {
    half8 df0 = vload8(1, src) - convert_half8(vload8(1, dest));
    half tail = src[3*3-1] - (half)dest[3*3-1];
    df0 *= df0 * (half8)SCORE_NORM;
    tail *= tail * SCORE_NORM;

    perColumnScore[0] += df0.s0 + df0.s1 + df0.s2;
    perColumnScore[1] += df0.s3 + df0.s4 + df0.s5;
    perColumnScore[2] += df0.s6 + df0.s7 + tail;
}

half inline appendScore3x1(const half *src, const __local uchar *dest) {
    half3 df0 = vload3(0, src) - convert_half3(vload3(0, dest));
    df0 *= df0 * (half3)SCORE_NORM;

    return df0.s0 + df0.s1 + df0.s2;
}

half inline getWindowColumnsScore(const half *src, const __local uchar *dest, half *perColumnScore) {
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
        case 11:
            appendScore11x11(src, dest, perColumnScore);
    }

    half scores = 0;
    for (int i = 0; i < WINDOW_SIZE; ++i) {
        scores += perColumnScore[i];
    }

    return scores;
}

half getPatchScore(half *patch) {
    switch (WINDOW_SIZE) {
        case 3:
            return getPatchScore3x3(patch);
        case 5:
            return getPatchScore5x5(patch);
        case 7:
            return getPatchScore7x7(patch);
        case 9:
            return getPatchScore9x9(patch);
        case 11:
            return getPatchScore11x11(patch);
    }
}

void inline loadPatch(const uchar *src, half *patch) {
    switch (WINDOW_SIZE) {
        case 3:
            loadPatch3x3(src, patch);
        case 5:
            loadPatch5x5(src, patch);
        case 7:
            loadPatch7x7(src, patch);
        case 9:
            loadPatch9x9(src, patch);
        case 11:
            loadPatch11x11(src, patch);
    }
}

void inline loadPatch1(const uchar *src, half *patch) {
    switch (WINDOW_SIZE) {
        case 3:
            loadPatch3x1(src, patch);
        case 5:
            loadPatch5x1(src, patch);
        case 7:
            loadPatch7x1(src, patch);
        case 9:
            loadPatch9x1(src, patch);
        case 11:
            loadPatch11x1(src, patch);
    }
}

half inline getWindowSingleColumnScore(const half *src, const __local uchar *dest) {
    switch (WINDOW_SIZE) {
        case 3:
            return appendScore3x1(src, dest);
        case 5:
            return appendScore5x1(src, dest);
        case 7:
            return appendScore7x1(src, dest);
        case 9:
            return appendScore9x1(src, dest);
        case 11:
            return appendScore11x1(src, dest);
    }
}

half inline interpolate(const half *costs, int i) {
    half C0 = 1.0 / costs[(i - 1) % 3];
    half C1 = 1.0 / costs[i % 3];
    half C2 = 1.0 / costs[(i + 1) % 3];

    return C1 > C0 && C1 > C2 ? (C0 - C2) / (2 * (C0 - 2 * C1 + C2)) : 0;
}

int getDisparityCandidates(
        half *patch,
        half *patchQuality,
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
        half wAvg,
        bool debug
        BC_ARG
) {
    __local const uchar *src = data1 + y + x * h;
    __local const uchar *dest = data2 + y + x * h;

    size_t dataSize = w * h;

    int disparityRange = maxDisparity - minDisparity;

    half perColumnScore[BATCH_SIZE + WINDOW_SIZE + 3] __attribute__ ((aligned (64)));

    int i = 0;

    half alpha = 2.0 / (wAvg + 1.0);

    half avgb[BATCH_SIZE];

    for (int b = 0; b < BATCH_SIZE; ++b) {
        avgb[b] = 10;
    }

    int nMatches = 1;
    int stepScale = 1;

    for (half d = minDisparity; d <= maxDisparity; d += step * stepScale) {
        stepScale = max((short)1, (short)(clz((short)0) - clz((short)fabs(d)) - 5));
        half cost = getWindowColumnsScore(patch, &dest[(int)d * h], perColumnScore);

        for (int b = 0; b < BATCH_SIZE; ++b) {
            int bw = b + WINDOW_SIZE;
            if (b < BATCH_SIZE - 1) {
                perColumnScore[bw] = getWindowSingleColumnScore(patch + bw * h, &dest[(int)d * h + bw * h]);
            }
            if (fabs(result[b] / DISPARITY_SCALE - d) > step && cost < costs[b]) {
                for (int k = nCandidates - 2; k >= 0; k--) {
                    result[(k + 1) * BATCH_SIZE + b] = result[k * BATCH_SIZE + b];
                    costs[(k + 1) * BATCH_SIZE + b] = costs[k * BATCH_SIZE + b];
                }
            }

//            int b0 = 0;
            int b0 = b;
            avgb[b0] = alpha * cost + (1 - alpha) * avgb[b0];
            half c = avgb[b0];

            bool needUpdate = patchQuality[b] > 0.1 && avgb[b0] < costs[b];

            costs[b] = fmin(costs[b], avgb[b0]);

            cost = cost + (-perColumnScore[b] + perColumnScore[bw]);

            result[b] = needUpdate
                    ? (short) d * DISPARITY_SCALE - (wAvg - 1) / 2 * step * DISPARITY_SCALE
                    : result[b];

            nMatches += needUpdate;
        }

        i++;
    }

    return nMatches;
}

void getDisparity(
        half * patch,
        __local uchar *data1,
        __local uchar *data2,
        int x,
        int y,
        int minDisparity,
        int maxDisparity,
        short* result,
        half* costs,
        int step,
        half wAvg,
        bool debug
        BC_ARG
) {
    __local const uchar *src = data1 + y + x * MAX_FRAGMENT_HEIGHT;
    __local const uchar *dest = data2 + y + x * MAX_FRAGMENT_HEIGHT;

    half costAvg[BATCH_SIZE * 3] __attribute__ ((aligned (64)));
    half perColumnScore[WINDOW_SIZE] __attribute__ ((aligned (64)));

    int i = 0;

    half stepScale = (int)WINDOW_SIZE / 4;

    for (half d = minDisparity; d <= maxDisparity && stepScale >= 1; d += step) {
        half cost = getWindowColumnsScore(patch, &dest[(int)d * MAX_FRAGMENT_HEIGHT], perColumnScore);
        costAvg[(i % 3)] = cost;
        bool needUpdate = cost < costs[0];
        costs[0] = fmin(costs[0], cost);
        result[0] = needUpdate
                ? (short) (d * DISPARITY_SCALE + interpolate(costAvg, i) * step * DISPARITY_SCALE)
                : result[0];
        i++;
    }
}

uchar convolve(uchar *frame, int x, int y, int w, int h, half *k, int k0) {
    int k2 = k0 / 2;
    half conv = 0;

    for (int i = -k2; i <= k2; ++i) {
        for (int j = -k2; j <= k2; ++j) {
            conv += (half)frame[clamp(y + i, 0, h) * w + clamp(x + j, 0, w)] * k[(i + k2) * k0 + j + k2];
        }
    }

    return clamp((int)conv, 0, 255);
}

void knorm(half *k, int k0) {
    half conv = 0;

    half knorm = 0;

    for (int i = 0; i < k0 * k0; ++i) {
        knorm += fabs(k[i]);
    }

//    knorm = 1 / knorm;

    for (int i = 0; i < k0 * k0; ++i) {
        k[i] /= knorm;
    }
}

half consistanceCheck(short d0, half cost, short *disparity, uchar *image, int w, short *suggestD) {
    half dl = disparity[-1];
    half du = disparity[-w];
    half dul = disparity[-w-1];

    if (d0 == 0 || dl == 0 || du == 0 || dul == 0) {
        *suggestD = d0;
        return cost;
    }

    half i0 = image[0];
    half il = image[-1];
    half iu = image[-w];
    half iul = image[-w-1];

    half covx = (du - dul) / (iu - iul + 0.5);
    half covy = (dl - dul) / (il - iul + 0.5);
    half dx = (i0 - il) * (covx) + dl;
    half dy = (i0 - iu) * (covy) + du;

    half dmin = fmin(fmin(dx, dy), 0);
    half dmax = fmax(fmax(dx, dy), 255);
    half dmean = (dx + dy) / 2;

    half consist = ((dmin > d0 || dmax < d0) * 0.9 + fabs(d0 - dmean) * 0.1 / 256);

    *suggestD = dmean;

    return consist;
}

__kernel void DisparityEvaluator(
        __global uchar* frame0,
        __global uchar* frame1,
        __global short* disparity,
        __global float* variance,
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

    int xLimit = min(x0 + H_GRANULE_SIZE, w - WINDOW_SIZE - BATCH_SIZE - 1);
    int yLimit = min(y0 + V_GRANULE_SIZE / 2, h - V_GRANULE_SIZE / 2 - MAX_FRAGMENT_HEIGHT - WINDOW_SIZE);

    int fragmentHeight = MAX_FRAGMENT_HEIGHT;
    fragmentHeight = min(fragmentHeight, h - 1 - y0);

    half kern[] = {
            0, 0,  1, 0, 0,
            0, 1,  3, 1, 0,
            1, 3, 16, 3, 1,
            0, 1,  3, 1, 0,
            0, 0,  1, 0, 0
    };
//    half kern[] = {
//            5,   8,  10,   8,  5,
//            8,  20,  40,  20,  8,
//            10, 40,  80,  40,  10,
//            8,  20,  40,  20,  8,
//            5,   8,  10,   8,  5
//    };

//    half boxKernel[] = {
//            1,  1,  1,  1,  1,
//            1,  1,  1,  1,  1,
//            1,  1,  1,  1,  1,
//            1,  1,  1,  1,  1,
//            1,  1,  1,  1,  1
//    };

    int ks = 5;
    knorm(kern, ks);
    int w2 = WINDOW_SIZE / 2;

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
                    uchar c0 = convolve(frame0, x, y + r, w, h, kern, ks);
                    uchar c1 = convolve(frame1, x, y + r, w, h, kern, ks);

                    pFrame0[r + x * MAX_FRAGMENT_HEIGHT] = c0;
                    pFrame1[r + x * MAX_FRAGMENT_HEIGHT] = c1;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        short disparities[BATCH_SIZE * N_CANDIDATES];
        half costs[BATCH_SIZE * N_CANDIDATES];
        half avgb[BATCH_SIZE];

        xLimit = min(xLimit, w - MIN_VALID_DISPARITY - WINDOW_SIZE);

        half initialAvg = 0;

        for (int x = x0; x < xLimit; x += BATCH_SIZE) {
            for (int b = 0; b < BATCH_SIZE; ++b) {
                disparities[b] = 0;
                costs[b] = 1e12;
            }

            for (int b = BATCH_SIZE; b < BATCH_SIZE * N_CANDIDATES; ++b) {
                disparities[b] = 0;
                costs[b] = 0;
            }

            const int maxDisparity = max(0, min(MAX_DISPARITY, (int) (w - (x + WINDOW_SIZE + 1))));

            half patchQuality[BATCH_SIZE];
            half patch[(BATCH_SIZE + WINDOW_SIZE + 3) * WINDOW_SIZE] __attribute__ ((aligned (64)));
            loadPatch(pFrame0 + x * MAX_FRAGMENT_HEIGHT, patch);
            patchQuality[0] = getPatchScore(patch);

            for (int b = 0; b < BATCH_SIZE - 1; ++b) {
                int bw = b + WINDOW_SIZE;
                loadPatch1(pFrame0 + (x + bw) * MAX_FRAGMENT_HEIGHT, patch + bw * MAX_FRAGMENT_HEIGHT);
            }

            for (int b = 0; b < BATCH_SIZE; b++) {
                patchQuality[b] = getPatchScore(patch + b * MAX_FRAGMENT_HEIGHT);
            }

            int nMatches = getDisparityCandidates(patch, patchQuality, pFrame0, pFrame1, x, 0, w, MAX_FRAGMENT_HEIGHT,
                         max(-x, -MAX_DISPARITY), 0, WINDOW_SIZE, 1, disparities, costs, N_CANDIDATES, BATCH_SIZE, 2, 4, false BC_PASS);

            for (int b = 0; b < BATCH_SIZE && nMatches > 0; ++b) {
                for (int i = 0; i < 1; ++i) {
                    short d = disparities[b + i * BATCH_SIZE] / DISPARITY_SCALE;

                    if (d == 0) {
                        continue;
                    }

                    getDisparity(patch + b * MAX_FRAGMENT_HEIGHT, pFrame0, pFrame1, x + b, 0,
                                 max(-x, d - 2), min(maxDisparity - b, d + 2), &disparities[b + i * BATCH_SIZE], &costs[b + i * BATCH_SIZE], 1, 1, false BC_PASS);

                }
            }

            for (int b = 0; b < BATCH_SIZE && nMatches > 0; ++b) {
                short dlrc = 0;
                half cost = 1e12;

                for (int i = 0; i < 1; ++i) {
                    short d = disparities[b + i * BATCH_SIZE] / DISPARITY_SCALE;

                    if (d == 0) {
                        continue;
                    }

                    int safeX = clamp(x + b + d, 0, xLimit);

                    loadPatch(pFrame1 + safeX * MAX_FRAGMENT_HEIGHT, patch);

                    getDisparity(patch, pFrame1, pFrame0, safeX, 0,
                                           max(-x - b, -d - 5), min(maxDisparity - b, -d + 5), &dlrc, &cost,
                                           2, 1, false BC_PASS);
                }

                half c0 =  fabs(costs[b]);
                half c1 = fabs((costs[b + BATCH_SIZE]));

                int resultOffset = (y + w2) * w + x + w2 + b;
                short d0 = abs(disparities[b]);
                short d1 = abs(disparities[b + BATCH_SIZE]);

                short d = d0;

                disparity[resultOffset] =
                     abs(d - dlrc) < 2 * DISPARITY_SCALE
                     && c1 / (c0 + EPS) > 1.35
                     && abs(d) < (MAX_DISPARITY - 8) * DISPARITY_SCALE
                     ? (short)(((half)d * c0 + (half)dlrc * cost) / (c0 + cost))
//                     ? (d * c1 + dlrc * cost) / (c1 + cost)
                     : 0;

//                variance[resultOffset] = frame0[resultOffset];
                variance[resultOffset] = c0;//((float)(abs(dlrc) - d)) / DISPARITY_SCALE;
            }
        }
    }

    BC_DUMP("(%d;%d)", x0, y0);
}
