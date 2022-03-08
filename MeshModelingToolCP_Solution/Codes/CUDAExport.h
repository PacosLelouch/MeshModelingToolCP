#pragma once

#include "CUDAExportCommon.h"

/// CUDA Utilities Export
namespace CUDAExport
{

    CUDA_LIB_IMPORT void hostMalloc(void** hstPtr, size_t size);
    CUDA_LIB_IMPORT void hostFree(void* hstPtr);

    CUDA_LIB_IMPORT void deviceMalloc(void** devPtr, size_t size);
    CUDA_LIB_IMPORT void deviceFree(void* devPtr);
    CUDA_LIB_IMPORT void deviceSync();

    CUDA_LIB_IMPORT void deviceToHostMemcpy(void* hstDst, void* devSrc, size_t size);
    CUDA_LIB_IMPORT void hostToHostMemcpy(void* hstDst, void* hstSrc, size_t size);
    CUDA_LIB_IMPORT void deviceToDeviceMemcpy(void* devDst, void* devSrc, size_t size);
    CUDA_LIB_IMPORT void hostToDeviceMemcpy(void* devDst, void* hstSrc, size_t size);
}

#include "CUDABufferUtil.h"

/// CUDA Custom Functions Export
namespace CUDAExport
{
    CUDA_LIB_IMPORT void deviceUpdatePositionByNormalPeriod(
        Buffer<glm::vec3> devPositions, const Buffer<glm::vec3> devRestPositions, const Buffer<glm::vec3> devNormals, 
        float time, float speedRate, float varyRate,
        int blockSizeX);

    CUDA_LIB_IMPORT void deviceUpdatePositionFromOriginPeriod(
        Buffer<glm::vec3> devPositions, const Buffer<glm::vec3> devRestPositions, 
        float time, float speedRate, float varyRate,
        int blockSizeX);
}
