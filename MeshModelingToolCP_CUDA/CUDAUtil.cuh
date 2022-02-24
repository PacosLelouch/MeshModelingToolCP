#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define STRINGIFY(x) TOSTRING(x)
#define TOSTRING(x) #x

namespace CUDAUtil 
{
inline int convert_SMV_to_cores(int major, int minor)
{
    // Taken from Nvidia helper_cuda.h to get the number of SM and cuda cores
    // Defines for GPU Architecture types (using the SM version to determine the
    // # of cores per SM
    typedef struct
    {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m =
                 // SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},  // Kepler Generation (SM 3.0) GK10x class
        {0x32, 192},  // Kepler Generation (SM 3.2) GK10x class
        {0x35, 192},  // Kepler Generation (SM 3.5) GK11x class
        {0x37, 192},  // Kepler Generation (SM 3.7) GK21x class
        {0x50, 128},  // Maxwell Generation (SM 5.0) GM10x class
        {0x52, 128},  // Maxwell Generation (SM 5.2) GM20x class
        {0x53, 128},  // Maxwell Generation (SM 5.3) GM20x class
        {0x60, 64},   // Pascal Generation (SM 6.0) GP100 class
        {0x61, 128},  // Pascal Generation (SM 6.1) GP10x class
        {0x62, 128},  // Pascal Generation (SM 6.2) GP10x class
        {0x70, 64},   // Volta Generation (SM 7.0) GV100 class
        {0x72, 64},  {0x75, 64}, {0x80, 64}, {0x86, 128}, {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    // If we don't find the values, we default use the previous one to run
    // properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

// CUDA_ERROR
inline void HandleError(cudaError_t err, const char* file, int line)
{
    // Error handling micro, wrap it around function whenever possible
    if (err != cudaSuccess) {
        printf("Line %d File %s\n", line, file);
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
#ifdef _WIN32
        system("pause");
#else
        exit(EXIT_FAILURE);
#endif
    }
}
}

#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))