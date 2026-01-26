#ifndef clion_defines_cl // pragma once
#define clion_defines_cl

#ifdef __CLION_IDE__

#define __kernel
#define __global
#define __local
#define __constant
#define __private

typedef float half;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long size_t;
typedef unsigned long uintptr_t;
typedef signed long intptr_t;
typedef signed long ptrdiff_t;

#define FIELD2(expr) expr s0; expr s1;
#define FIELD3(expr) FIELD2(expr) expr s2;
#define FIELD4(expr) FIELD3(expr) expr s3;
#define FIELD5(expr) FIELD4(expr) expr s4;
#define FIELD6(expr) FIELD5(expr) expr s5;
#define FIELD7(expr) FIELD6(expr) expr s6;
#define FIELD8(expr) FIELD7(expr) expr s7;
#define FIELD9(expr) FIELD8(expr) expr s8;
#define FIELD10(expr) FIELD9(expr) expr s9;
#define FIELD11(expr) FIELD10(expr) expr sA;
#define FIELD12(expr) FIELD11(expr) expr sB;
#define FIELD13(expr) FIELD12(expr) expr sC;
#define FIELD14(expr) FIELD13(expr) expr sD;
#define FIELD15(expr) FIELD14(expr) expr sE;
#define FIELD16(expr) FIELD15(expr) expr sF;

#define VECTOR_1FN(T, fn) \
T ## 2 fn(T ## 2); \
T ## 3 fn(T ## 3); \
T ## 4 fn(T ## 4); \
T ## 8 fn(T ## 8); \
T ## 16 fn(T ## 16);

#define VECTOR_2FN(T, TI, fn) \
TI ## 2 fn(T ## 2, T ## 2); \
TI ## 3 fn(T ## 3, T ## 3); \
TI ## 4 fn(T ## 4, T ## 4); \
TI ## 8 fn(T ## 8, T ## 8); \
TI ## 16 fn(T ## 16, T ## 16);

#define VECTOR_VLOADSTORE(T) \
T ##  2 vload ##  2(size_t offset, const T *p); \
T ##  3 vload ##  3(size_t offset, const T *p); \
T ##  4 vload ##  4(size_t offset, const T *p); \
T ##  8 vload ##  8(size_t offset, const T *p); \
T ## 16 vload ## 16(size_t offset, const T *p); \
void vstore ##  2(T ##  2 data, size_t offset, const T *p); \
void vstore ##  3(T ##  3 data, size_t offset, const T *p); \
void vstore ##  4(T ##  4 data, size_t offset, const T *p); \
void vstore ##  8(T ##  8 data, size_t offset, const T *p); \
void vstore ## 16(T ## 16 data, size_t offset, const T *p);

#define VECTOR_CONVERT(T, T2) \
T ##  2 convert_ ## T ##  2(T2 ##  2); \
T ##  3 convert_ ## T ##  3(T2 ##  3); \
T ##  4 convert_ ## T ##  4(T2 ##  4); \
T ##  8 convert_ ## T ##  8(T2 ##  8); \
T ## 16 convert_ ## T ## 16(T2 ## 16);

#define VECTOR_CONVERT_ALL(T) \
VECTOR_CONVERT(T, char) \
VECTOR_CONVERT(T, uchar) \
VECTOR_CONVERT(T, short) \
VECTOR_CONVERT(T, ushort) \
VECTOR_CONVERT(T, int) \
VECTOR_CONVERT(T, uint) \
VECTOR_CONVERT(T, long) \
VECTOR_CONVERT(T, ulong) \
VECTOR_CONVERT(T, half) \
VECTOR_CONVERT(T, float) \
VECTOR_CONVERT(T, double)

#define VECTOR_TYPE(T, TI) \
struct T ## 2 { FIELD2(T); T &operator[](size_t); }; \
struct T ## 3 { FIELD3(T); T &operator[](size_t); }; \
struct T ## 4 { FIELD4(T); T &operator[](size_t); }; \
struct T ## 8 { FIELD8(T); T &operator[](size_t); }; \
struct T ## 16 { FIELD16(T); T &operator[](size_t); }; \
VECTOR_2FN(T, TI, operator==) \
VECTOR_2FN(T, TI, operator>) \
VECTOR_2FN(T, TI, operator<) \
VECTOR_2FN(T, TI, operator>=) \
VECTOR_2FN(T, TI, operator<=) \
VECTOR_2FN(T, TI, operator&&) \
VECTOR_2FN(T, TI, operator||) \
VECTOR_2FN(T, TI, operator<<) \
VECTOR_2FN(T, TI, operator>>) \
VECTOR_2FN(T, T, operator-) \
VECTOR_2FN(T, T, operator+) \
VECTOR_2FN(T, T, operator*) \
VECTOR_2FN(T, T, operator/) \
VECTOR_2FN(T, TI, operator<<=) \
VECTOR_2FN(T, TI, operator>>=) \
VECTOR_2FN(T, T, operator-=) \
VECTOR_2FN(T, T, operator+=) \
VECTOR_2FN(T, T, operator*=) \
VECTOR_2FN(T, T, operator/=) \
VECTOR_1FN(T, operator++) \
VECTOR_1FN(T, operator--)

VECTOR_TYPE(char, char);
VECTOR_TYPE(uchar, uchar);
VECTOR_TYPE(short, short);
VECTOR_TYPE(ushort, ushort);
VECTOR_TYPE(int, int);
VECTOR_TYPE(uint, uint);
VECTOR_TYPE(long, long);
VECTOR_TYPE(ulong, ulong);
VECTOR_TYPE(half, short);
VECTOR_TYPE(float, int);
VECTOR_TYPE(double, long);

VECTOR_VLOADSTORE(char);
VECTOR_VLOADSTORE(uchar);
VECTOR_VLOADSTORE(short);
VECTOR_VLOADSTORE(ushort);
VECTOR_VLOADSTORE(int);
VECTOR_VLOADSTORE(uint);
VECTOR_VLOADSTORE(long);
VECTOR_VLOADSTORE(ulong);
VECTOR_VLOADSTORE(float);
VECTOR_VLOADSTORE(double);

VECTOR_CONVERT_ALL(char);
VECTOR_CONVERT_ALL(uchar);
VECTOR_CONVERT_ALL(short);
VECTOR_CONVERT_ALL(ushort);
VECTOR_CONVERT_ALL(int);
VECTOR_CONVERT_ALL(uint);
VECTOR_CONVERT_ALL(long);
VECTOR_CONVERT_ALL(ulong);
VECTOR_CONVERT_ALL(float);
VECTOR_CONVERT_ALL(double);


// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/commonFunctions.html
#define gentype float
#define igentype int
gentype		clamp		(gentype x, float minval, float maxval);
igentype		clamp		(igentype x, igentype minval, igentype maxval);
gentype		degrees		(gentype radians);
gentype		max			(gentype x, gentype y);
igentype		max			(igentype x, igentype y);
gentype		min			(gentype x, gentype y);
igentype		min			(igentype x, igentype y);
gentype		fmax		(gentype x, gentype y);
gentype		fmin		(gentype x, gentype y);
gentype		mix			(gentype x, gentype y, gentype a);
gentype		radians		(gentype degrees);
gentype		sign		(gentype x);
gentype		smoothstep	(gentype edge0, gentype edge1, gentype x);
gentype		step		(gentype edge, gentype x);
igentype	rint		(gentype x);
#undef gentype
#undef igentype

// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/barrier.html
enum	cl_mem_fence_flags
{
    CLK_LOCAL_MEM_FENCE,
    CLK_GLOBAL_MEM_FENCE
};
void	barrier(cl_mem_fence_flags flags);

// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/workItemFunctions.html
uint	get_work_dim		();
size_t	get_global_size		(uint dimindx);
size_t	get_global_id		(uint dimindx);
size_t	get_local_size		(uint dimindx);
size_t	get_local_id		(uint dimindx);
size_t	get_num_groups		(uint dimindx);
size_t	get_group_id		(uint dimindx);
size_t	get_global_offset	(uint dimindx);

#ifndef STATIC_KEYWORD
#define STATIC_KEYWORD static
#endif

#endif

#endif // pragma once