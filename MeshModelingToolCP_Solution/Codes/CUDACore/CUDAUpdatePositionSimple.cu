#include "ExportCommon.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "glm/glm.hpp"
#include "CUDABufferUtil.h"
//#include "CUDAExport.h"


namespace CUDAExport
{
    __global__ void kernUpdatePositionByNormalPeriod(
        Buffer<glm::vec3> devPositions, const Buffer<glm::vec3> devRestPositions, const Buffer<glm::vec3> devNormals, 
        float time, float speedRate, float varyRate)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index > devPositions.count)
        {
            return;
        }
        devPositions[index] = devRestPositions[index] + devNormals[index] * (sin(time * speedRate) * varyRate);
    }

    DLL_EXPORT void deviceUpdatePositionByNormalPeriod(
        Buffer<glm::vec3> devPositions, const Buffer<glm::vec3> devRestPositions, const Buffer<glm::vec3> devNormals, 
        float time, float speedRate, float varyRate,
        int blockSizeX)
    {
        int gridSizeX = DivUp(devPositions.count, blockSizeX);
        kernUpdatePositionByNormalPeriod<<<gridSizeX, blockSizeX>>>(devPositions, devRestPositions, devNormals, time, speedRate, varyRate);
    }

    __global__ void kernUpdatePositionFromOriginPeriod(
        Buffer<glm::vec3> devPositions, const Buffer<glm::vec3> devRestPositions, 
        float time, float speedRate, float varyRate)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index > devPositions.count)
        {
            return;
        }
        glm::vec3 devRestPos = devRestPositions[index];
        devPositions[index] = devRestPos + glm::normalize(devRestPos) * (sin(time * speedRate) * varyRate);
    }

    DLL_EXPORT void deviceUpdatePositionFromOriginPeriod(
        Buffer<glm::vec3> devPositions, const Buffer<glm::vec3> devRestPositions, 
        float time, float speedRate, float varyRate, 
        int blockSizeX)
    {
        int gridSizeX = DivUp(devPositions.count, blockSizeX);
        kernUpdatePositionFromOriginPeriod<<<gridSizeX, blockSizeX>>>(devPositions, devRestPositions, time, speedRate, varyRate);
    }
}