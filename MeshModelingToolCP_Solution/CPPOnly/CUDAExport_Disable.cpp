#include "ExportCommon.h"
#include <memory>
#include <iostream>

namespace CUDAExport
{    
    MAYA_EXPORT void hostMalloc(void** hstPtr, size_t size)
    {
        if (hstPtr)
        {
            *hstPtr = malloc(size);
        }
    }

    MAYA_EXPORT void hostFree(void* hstPtr)
    {
        if (hstPtr)
        {
            free(hstPtr);
        }
    }

    MAYA_EXPORT void deviceMalloc(void** devPtr, size_t size)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    MAYA_EXPORT void deviceFree(void* devPtr)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    MAYA_EXPORT void deviceSync()
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    MAYA_EXPORT void deviceToHostMemcpy(void* hstDst, void* devSrc, size_t size)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    MAYA_EXPORT void hostToHostMemcpy(void* hstDst, void* hstSrc, size_t size)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    MAYA_EXPORT void deviceToDeviceMemcpy(void* devDst, void* devSrc, size_t size)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }

    MAYA_EXPORT void hostToDeviceMemcpy(void* devDst, void* hstSrc, size_t size)
    {
        CUDA_DISABLE_ERROR();
        exit(1);
    }
}