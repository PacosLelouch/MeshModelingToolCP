#pragma once

#include "ExportCommon.h"
#include "CUDABufferUtil.h"

/// CUDA Utilities Export
namespace CUDAExport
{

    DLL_IMPORT void hostMalloc(void** hstPtr, size_t size);
    DLL_IMPORT void hostFree(void* hstPtr);

    DLL_IMPORT void deviceMalloc(void** devPtr, size_t size);
    DLL_IMPORT void deviceFree(void* devPtr);
    DLL_IMPORT void deviceSync();

    DLL_IMPORT void deviceToHostMemcpy(void* hstDst, void* devSrc, size_t size);
    DLL_IMPORT void hostToHostMemcpy(void* hstDst, void* hstSrc, size_t size);
    DLL_IMPORT void deviceToDeviceMemcpy(void* devDst, void* devSrc, size_t size);
    DLL_IMPORT void hostToDeviceMemcpy(void* devDst, void* hstSrc, size_t size);
}


/// CUDA Custom Functions Export
namespace CUDAExport
{
    DLL_IMPORT void deviceUpdatePositionByNormalPeriod(
        Buffer<glm::vec3> devPositions, const Buffer<glm::vec3> devRestPositions, const Buffer<glm::vec3> devNormals, 
        float time, float speedRate, float varyRate,
        int blockSizeX);

    DLL_IMPORT void deviceUpdatePositionFromOriginPeriod(
        Buffer<glm::vec3> devPositions, const Buffer<glm::vec3> devRestPositions, 
        float time, float speedRate, float varyRate,
        int blockSizeX);
}
