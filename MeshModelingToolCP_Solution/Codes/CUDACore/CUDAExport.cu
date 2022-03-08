#include "ExportCommon.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace CUDAExport
{    
    DLL_EXPORT void hostMalloc(void** hstPtr, size_t size)
    {
        cudaMallocHost(hstPtr, size);
    }

    DLL_EXPORT void hostFree(void* hstPtr)
    {
        cudaFreeHost(hstPtr);
    }

    DLL_EXPORT void deviceMalloc(void** devPtr, size_t size)
    {
        cudaMalloc(devPtr, size);
    }

    DLL_EXPORT void deviceFree(void* devPtr)
    {
        cudaFree(devPtr);
    }

    DLL_EXPORT void deviceSync()
    {
        cudaDeviceSynchronize();
    }

    DLL_EXPORT void deviceToHostMemcpy(void* hstDst, void* devSrc, size_t size)
    {
        cudaMemcpy(hstDst, devSrc, size, cudaMemcpyDeviceToHost);
    }

    DLL_EXPORT void hostToHostMemcpy(void* hstDst, void* hstSrc, size_t size)
    {
        cudaMemcpy(hstDst, hstSrc, size, cudaMemcpyHostToHost);
    }

    DLL_EXPORT void deviceToDeviceMemcpy(void* devDst, void* devSrc, size_t size)
    {
        cudaMemcpy(devDst, devSrc, size, cudaMemcpyDeviceToDevice);
    }

    DLL_EXPORT void hostToDeviceMemcpy(void* devDst, void* hstSrc, size_t size)
    {
        cudaMemcpy(devDst, hstSrc, size, cudaMemcpyHostToDevice);
    }
}