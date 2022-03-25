#include "CUDATest.h"
#include "CUDAUtil.cuh"
#include <stdio.h>
#include <string.h>
#include <string>

#if defined _WIN32 || defined _WIN64
#include <Windows.h>

#else
#include <stdio.h>
#endif

using namespace CUDAUtil;

namespace CUDATest 
{
__global__ static void get_cude_arch_k(int* d_arch)
{

#if defined(__CUDA_ARCH__)
    *d_arch = __CUDA_ARCH__;
#else
    *d_arch = 0;
#endif
}

inline int cuda_arch()
{
    int* d_arch = 0;
    CUDA_ERROR(cudaMalloc((void**)&d_arch, sizeof(int)));
    get_cude_arch_k<<<1, 1>>>(d_arch);
    int h_arch = 0;
    CUDA_ERROR(
        cudaMemcpy(&h_arch, d_arch, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_arch);
    return h_arch;
}

template<typename TCHAR>
static TCHAR* getFrac(TCHAR* dst, double f, int precision) 
{
    //int result = 0;
    double f1 = f;
    int i1 = static_cast<int>(f);
    size_t cursor = 0;
    for (int _ = 0; _ < precision; ++_) {
        f1 = (f1 - (double)i1) * 10.;
        dst[cursor++] = '0' + (int)f1;
        i1 = static_cast<int>(f1);
    }
    dst[cursor] = '\0';
    return dst;
}

#ifdef UNICODE
using CHAR_T = WCHAR;
using LPSTR_T = LPWSTR;
#define mbstowcs_t(A,B,C) mbstowcs(A,B,C)
#else
using CHAR_T = CHAR;
using LPSTR_T = LPSTR;
#define mbstowcs_t(A,B,C) strcpy(A,B)
#endif

cudaDeviceProp cuda_query(const int dev, bool quiet = false, std::string* outStr = nullptr)
{
    cudaDeviceProp devProp;
    CHAR_T message[16384]{ 0 };
    //memset(message, 0, sizeof(message));
    LPSTR_T cursor = message;
    //cursor += wsprintf(cursor, TEXT("----------------CUDATestFunction------------------\n"));
    // Various query about the device we are using
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) 
    {
        cursor += wsprintf(cursor, 
            TEXT("cuda_query() device count = 0 i.e., there is not") 
            TEXT(" a CUDA-supported GPU!!!\n"));
    }
    else
    {
        cudaSetDevice(dev);

        CUDA_ERROR(cudaGetDeviceProperties(&devProp, dev));

        cursor += wsprintf(cursor, TEXT("Total number of device: %d\n"), deviceCount);
        cursor += wsprintf(cursor, TEXT("Using device Number: %d\n"), dev);
        CHAR_T devName[256]{ 0 };
        mbstowcs_t(devName, devProp.name, strlen(devProp.name));
        cursor += wsprintf(cursor, TEXT("Device name: %s\n"), devName);
        cursor += wsprintf(cursor, TEXT("Compute Capability: %d.%d\n"), (int)devProp.major,
            (int)devProp.minor);
        float gmem = (float)devProp.totalGlobalMem / 1048576.0f;
        CHAR_T frac0[16]{ 0 };
        CHAR_T frac1[16]{ 0 };
        cursor += wsprintf(cursor, TEXT("Total amount of global memory (MB): %d.%s\n"), 
            (int)gmem, getFrac(frac0, gmem, 1));
        cursor += wsprintf(cursor, TEXT("%d Multiprocessors, %d CUDA Cores/MP: %d CUDA Cores\n"), 
            devProp.multiProcessorCount,
            convert_SMV_to_cores(devProp.major, devProp.minor),
            convert_SMV_to_cores(devProp.major, devProp.minor) *
            devProp.multiProcessorCount);
        float GPUMaxClockRateMHz = devProp.clockRate * 1e-3f;
        float GPUMaxClockRateGHz = devProp.clockRate * 1e-6f;
        cursor += wsprintf(cursor, TEXT("GPU Max Clock rate: %d.%s MHz (%d.%s GHz)\n"), 
            (int)GPUMaxClockRateMHz, getFrac(frac0, GPUMaxClockRateMHz, 2),
            (int)GPUMaxClockRateGHz, getFrac(frac1, GPUMaxClockRateGHz, 2));
        float memMaxClockRateMHz = devProp.memoryClockRate * 1e-3f;
        cursor += wsprintf(cursor, TEXT("Memory Clock rate: %d.%s Mhz\n"), 
            (int)memMaxClockRateMHz, getFrac(frac0, memMaxClockRateMHz, 2));
        cursor += wsprintf(cursor, TEXT("Memory Bus Width:  %d-bit\n"), devProp.memoryBusWidth);
        const double maxBW = 2.0 * devProp.memoryClockRate *
            (devProp.memoryBusWidth / 8.0) / 1.0E6;

        cursor += wsprintf(cursor, TEXT("Peak Memory Bandwidth: %d.%s(GB/s)\n"), (int)maxBW, getFrac(frac0, maxBW, 2));
        cursor += wsprintf(cursor, TEXT("Kernels compiled for compute capability: %d"), 
            cuda_arch());
    }

    if (!quiet) {
#if defined _WIN32 || defined _WIN64
        MessageBox(NULL, message, TEXT("Third Party Plugin CUDA Test"), MB_OK);
#else
        wprintf(TEXT("%s"), message);
#endif
    }

    if (outStr)
    {
        CHAR messageANSI[16384]{ 0 };
        if (CharToOem(message, messageANSI))
        {
            outStr->assign(messageANSI);
        }
    }

    return devProp;
}

void CUDATest::CUDATestFunction(bool quiet, std::string* outStr) 
{
    cuda_query(0, quiet, outStr);
}
}