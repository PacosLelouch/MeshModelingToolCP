#include "CUDAExportCommon.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace CUDAExport
{    
    CUDA_LIB_EXPORT void hostMalloc(void** hstPtr, size_t size)
    {
        cudaMallocHost(hstPtr, size);
    }

    CUDA_LIB_EXPORT void hostFree(void* hstPtr)
    {
        cudaFreeHost(hstPtr);
    }

    CUDA_LIB_EXPORT void deviceMalloc(void** devPtr, size_t size)
    {
        cudaMalloc(devPtr, size);
    }

    CUDA_LIB_EXPORT void deviceFree(void* devPtr)
    {
        cudaFree(devPtr);
    }

    CUDA_LIB_EXPORT void deviceSync()
    {
        cudaDeviceSynchronize();
    }

    CUDA_LIB_EXPORT void deviceToHostMemcpy(void* hstDst, void* devSrc, size_t size)
    {
        cudaMemcpy(hstDst, devSrc, size, cudaMemcpyDeviceToHost);
    }

    CUDA_LIB_EXPORT void hostToHostMemcpy(void* hstDst, void* hstSrc, size_t size)
    {
        cudaMemcpy(hstDst, hstSrc, size, cudaMemcpyHostToHost);
    }

    CUDA_LIB_EXPORT void deviceToDeviceMemcpy(void* devDst, void* devSrc, size_t size)
    {
        cudaMemcpy(devDst, devSrc, size, cudaMemcpyDeviceToDevice);
    }

    CUDA_LIB_EXPORT void hostToDeviceMemcpy(void* devDst, void* hstSrc, size_t size)
    {
        cudaMemcpy(devDst, hstSrc, size, cudaMemcpyHostToDevice);
    }
}