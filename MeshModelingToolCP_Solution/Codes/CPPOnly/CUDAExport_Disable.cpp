#include "CUDAExportCommon.h"
#include <memory>
#include <iostream>

namespace CUDAExport
{    
    CUDA_LIB_EXPORT void hostMalloc(void** hstPtr, size_t size)
    {
        if (hstPtr)
        {
            *hstPtr = malloc(size);
        }
    }

    CUDA_LIB_EXPORT void hostFree(void* hstPtr)
    {
        if (hstPtr)
        {
            free(hstPtr);
        }
    }

    CUDA_LIB_EXPORT void deviceMalloc(void** devPtr, size_t size)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    CUDA_LIB_EXPORT void deviceFree(void* devPtr)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    CUDA_LIB_EXPORT void deviceSync()
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    CUDA_LIB_EXPORT void deviceToHostMemcpy(void* hstDst, void* devSrc, size_t size)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    CUDA_LIB_EXPORT void hostToHostMemcpy(void* hstDst, void* hstSrc, size_t size)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    CUDA_LIB_EXPORT void deviceToDeviceMemcpy(void* devDst, void* devSrc, size_t size)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    CUDA_LIB_EXPORT void hostToDeviceMemcpy(void* devDst, void* hstSrc, size_t size)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }
}