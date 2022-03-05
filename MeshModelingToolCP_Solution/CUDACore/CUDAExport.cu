#include "ExportCommon.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace CUDAExport
{    
    MAYA_EXPORT void hostMalloc(void** hstPtr, size_t size)
    {
        cudaMallocHost(hstPtr, size);
    }

    MAYA_EXPORT void hostFree(void* hstPtr)
    {
        cudaFreeHost(hstPtr);
    }

    MAYA_EXPORT void deviceMalloc(void** devPtr, size_t size)
    {
        cudaMalloc(devPtr, size);
    }

    MAYA_EXPORT void deviceFree(void* devPtr)
    {
        cudaFree(devPtr);
    }

    MAYA_EXPORT void deviceSync()
    {
        cudaDeviceSynchronize();
    }

    MAYA_EXPORT void deviceToHostMemcpy(void* hstDst, void* devSrc, size_t size)
    {
        cudaMemcpy(hstDst, devSrc, size, cudaMemcpyDeviceToHost);
    }

    MAYA_EXPORT void hostToHostMemcpy(void* hstDst, void* hstSrc, size_t size)
    {
        cudaMemcpy(hstDst, hstSrc, size, cudaMemcpyHostToHost);
    }

    MAYA_EXPORT void deviceToDeviceMemcpy(void* devDst, void* devSrc, size_t size)
    {
        cudaMemcpy(devDst, devSrc, size, cudaMemcpyDeviceToDevice);
    }

    MAYA_EXPORT void hostToDeviceMemcpy(void* devDst, void* hstSrc, size_t size)
    {
        cudaMemcpy(devDst, hstSrc, size, cudaMemcpyHostToDevice);
    }
}