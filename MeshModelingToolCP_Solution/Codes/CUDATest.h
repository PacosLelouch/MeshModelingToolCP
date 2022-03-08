#pragma once
#include <string>

namespace CUDATest 
{
class CUDATest
{
public:
    static void CUDATestFunction(bool quiet = false, std::string* outStr = nullptr);
};
}