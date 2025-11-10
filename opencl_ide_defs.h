#ifndef clion_defines_cl // pragma once
#define clion_defines_cl

#ifdef __CLION_IDE__

#define __kernel
#define __global
#define __local
#define __constant
#define __private

#define half float
typedef unsigned char uchar;
typedef unsigned char uchar2;
typedef unsigned char uchar3;
typedef unsigned char uchar4;
typedef unsigned char uchar8;
typedef unsigned char uchar16;

typedef unsigned char char2;
typedef unsigned char char3;
typedef unsigned char char4;
typedef unsigned char char8;
typedef unsigned char char16;

typedef signed short short2;
typedef signed short short3;
typedef signed short short4;
typedef signed short short8;
typedef signed short short16;

typedef unsigned short ushort;
typedef unsigned short ushort2;
typedef unsigned short ushort3;
typedef unsigned short ushort4;
typedef unsigned short ushort8;
typedef unsigned short ushort16;

typedef signed int int2;
typedef signed int int3;
typedef signed int int4;
typedef signed int int8;
typedef signed int int16;

typedef unsigned int uint;
typedef unsigned int uint2;
typedef unsigned int uint3;
typedef unsigned int uint4;
typedef unsigned int uint8;
typedef unsigned int uint16;

typedef signed long long2;
typedef signed long long3;
typedef signed long long4;
typedef signed long long8;
typedef signed long long16;

typedef unsigned long ulong;
typedef unsigned long ulong2;
typedef unsigned long ulong3;
typedef unsigned long ulong4;
typedef unsigned long ulong8;
typedef unsigned long ulong16;

typedef float float2;
typedef float float3;
typedef float float4;
typedef float float8;
typedef float float16;

typedef double double2;
typedef double double3;
typedef double double4;
typedef double double8;
typedef double double16;

typedef unsigned long size_t;
typedef unsigned long uintptr_t;

// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/commonFunctions.html
#define gentype float
gentype		clamp		(gentype x, float minval, float maxval);
gentype		degrees		(gentype radians);
gentype		max			(gentype x, gentype y);
gentype		min			(gentype x, gentype y);
gentype		fmax		(gentype x, gentype y);
gentype		fmin		(gentype x, gentype y);
gentype		mix			(gentype x, gentype y, gentype a);
gentype		radians		(gentype degrees);
gentype		sign		(gentype x);
gentype		smoothstep	(gentype edge0, gentype edge1, gentype x);
gentype		step		(gentype edge, gentype x);
#undef gentype

// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/barrier.html
enum	cl_mem_fence_flags
{
    CLK_LOCAL_MEM_FENCE,
    CLK_GLOBAL_MEM_FENCE
};
void	barrier(cl_mem_fence_flags flags);

// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/vectorDataLoadandStoreFunctions.html
#define gentype float
#define gentypen float4
gentypen	vload4			(size_t offset, const gentype *p);
void		vstore4			(gentypen data, size_t offset, gentype *p);
void		vstore4			(gentypen data, size_t offset, gentype *p);
#undef gentypen
#undef gentype
float		vload_half		(size_t offset, const half *p);
float4		vload_half4		(size_t offset, const half *p);
void		vstore_half		(float data, size_t offset, half *p);
void		vstore_half4	(float4 data, size_t offset, half *p);
float4		vloada_half4	(size_t offset, const half *p);
void		vstorea_half4	(float4 data, size_t offset, half *p);

// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/workItemFunctions.html
uint	get_work_dim		();
size_t	get_global_size		(uint dimindx);
size_t	get_global_id		(uint dimindx);
size_t	get_local_size		(uint dimindx);
size_t	get_local_id		(uint dimindx);
size_t	get_num_groups		(uint dimindx);
size_t	get_group_id		(uint dimindx);
size_t	get_global_offset	(uint dimindx);

uchar2	vload2			(size_t offset, const uchar *p);
char2	vload2			(size_t offset, const char *p);
ushort2	vload2			(size_t offset, const ushort *p);
short2	vload2			(size_t offset, const short *p);
int2	vload2			(size_t offset, const int *p);
uint2	vload2			(size_t offset, const uint *p);
long2	vload2			(size_t offset, const long *p);
ulong2	vload2			(size_t offset, const ulong *p);
float2	vload2			(size_t offset, const float *p);
double2	vload2			(size_t offset, const double *p);

uchar4	vload4			(size_t offset, const uchar *p);
char4	vload4			(size_t offset, const char *p);
ushort4	vload4			(size_t offset, const ushort *p);
short4	vload4			(size_t offset, const short *p);
int4	vload4			(size_t offset, const int *p);
uint4	vload4			(size_t offset, const uint *p);
long4	vload4			(size_t offset, const long *p);
ulong4	vload4			(size_t offset, const ulong *p);
float4	vload4			(size_t offset, const float *p);
double4	vload4			(size_t offset, const double *p);

uchar8	vload8			(size_t offset, const uchar *p);
char8	vload8			(size_t offset, const char *p);
ushort8	vload8			(size_t offset, const ushort *p);
short8	vload8			(size_t offset, const short *p);
int8	vload8			(size_t offset, const int *p);
uint8	vload8			(size_t offset, const uint *p);
long8	vload8			(size_t offset, const long *p);
ulong8	vload8			(size_t offset, const ulong *p);
float8	vload8			(size_t offset, const float *p);
double8	vload8			(size_t offset, const double *p);

uchar16	vload16			(size_t offset, const uchar *p);
char16	vload16			(size_t offset, const char *p);
ushort16	vload16			(size_t offset, const ushort *p);
short16	vload16			(size_t offset, const short *p);
int16	vload16			(size_t offset, const int *p);
uint16	vload16			(size_t offset, const uint *p);
long16	vload16			(size_t offset, const long *p);
ulong16	vload16			(size_t offset, const ulong *p);
float16	vload16			(size_t offset, const float *p);
double16	vload16			(size_t offset, const double *p);

void vstore2			(uchar2 data, size_t offset, const uchar *p);
void vstore2			(char2 data, size_t offset, const char *p);
void vstore2			(ushort2 data, size_t offset, const ushort *p);
void vstore2			(short2 data, size_t offset, const short *p);
void vstore2			(int2 data, size_t offset, const int *p);
void vstore2			(uint2 data, size_t offset, const uint *p);
void vstore2			(long2 data, size_t offset, const long *p);
void vstore2			(ulong2 data, size_t offset, const ulong *p);
void vstore2			(float2 data, size_t offset, const float *p);
void vstore2			(double2 data, size_t offset, const double *p);

void vstore4			(uchar4 data, size_t offset, const uchar *p);
void vstore4			(char4 data, size_t offset, const char *p);
void vstore4			(ushort4 data, size_t offset, const ushort *p);
void vstore4			(short4 data, size_t offset, const short *p);
void vstore4			(int4 data, size_t offset, const int *p);
void vstore4			(uint4 data, size_t offset, const uint *p);
void vstore4			(long4 data, size_t offset, const long *p);
void vstore4			(ulong4 data, size_t offset, const ulong *p);
void vstore4			(float4 data, size_t offset, const float *p);
void vstore4			(double4 data, size_t offset, const double *p);

void vstore8			(uchar8 data, size_t offset, const uchar *p);
void vstore8			(char8 data, size_t offset, const char *p);
void vstore8			(ushort8 data, size_t offset, const ushort *p);
void vstore8			(short8 data, size_t offset, const short *p);
void vstore8			(int8 data, size_t offset, const int *p);
void vstore8			(uint8 data, size_t offset, const uint *p);
void vstore8			(long8 data, size_t offset, const long *p);
void vstore8			(ulong8 data, size_t offset, const ulong *p);
void vstore8			(float8 data, size_t offset, const float *p);
void vstore8			(double8 data, size_t offset, const double *p);

void vstore16			(uchar16 data, size_t offset, const uchar *p);
void vstore16			(char16 data, size_t offset, const char *p);
void vstore16			(ushort16 data, size_t offset, const ushort *p);
void vstore16			(short16 data, size_t offset, const short *p);
void vstore16			(int16 data, size_t offset, const int *p);
void vstore16			(uint16 data, size_t offset, const uint *p);
void vstore16			(long16 data, size_t offset, const long *p);
void vstore16			(ulong16 data, size_t offset, const ulong *p);
void vstore16			(float16 data, size_t offset, const float *p);
void vstore16			(double16 data, size_t offset, const double *p);

#ifndef STATIC_KEYWORD
#define STATIC_KEYWORD static
#endif

#endif

#endif // pragma once