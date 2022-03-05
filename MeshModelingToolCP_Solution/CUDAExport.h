#pragma once

#include "ExportCommon.h"
#include "CUDABufferUtil.h"

/// CUDA Utilities Export
namespace CUDAExport
{

    MAYA_IMPORT void hostMalloc(void** hstPtr, size_t size);
    MAYA_IMPORT void hostFree(void* hstPtr);

    MAYA_IMPORT void deviceMalloc(void** devPtr, size_t size);
    MAYA_IMPORT void deviceFree(void* devPtr);
    MAYA_IMPORT void deviceSync();

    MAYA_IMPORT void deviceToHostMemcpy(void* hstDst, void* devSrc, size_t size);
    MAYA_IMPORT void hostToHostMemcpy(void* hstDst, void* hstSrc, size_t size);
    MAYA_IMPORT void deviceToDeviceMemcpy(void* devDst, void* devSrc, size_t size);
    MAYA_IMPORT void hostToDeviceMemcpy(void* devDst, void* hstSrc, size_t size);
}


/// CUDA Custom Functions Export
namespace CUDAExport
{
    MAYA_IMPORT void deviceUpdatePositionByNormalPeriod(
        Buffer<glm::vec3> devPositions, const Buffer<glm::vec3> devRestPositions, const Buffer<glm::vec3> devNormals, 
        float time, float speedRate, float varyRate,
        int blockSizeX);

    MAYA_IMPORT void deviceUpdatePositionFromOriginPeriod(
        Buffer<glm::vec3> devPositions, const Buffer<glm::vec3> devRestPositions, 
        float time, float speedRate, float varyRate,
        int blockSizeX);
}
