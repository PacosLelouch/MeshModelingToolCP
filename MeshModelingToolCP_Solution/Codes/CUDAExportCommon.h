#pragma once

#undef CUDA_LIB_IMPORT
#undef CUDA_LIB_EXPORT

//#if defined _WIN32 || defined _WIN64
//#define CUDA_LIB_IMPORT __declspec(dllimport)
//#elif defined __linux__
//#define CUDA_LIB_IMPORT __attribute__((visibility("default")))
//#else
//#define CUDA_LIB_IMPORT
//#endif
//
//#if defined _WIN32 || defined _WIN64
//#define CUDA_LIB_EXPORT __declspec(dllexport)
//#else
//#endif
//
//#ifndef CUDA_LIB_EXPORT
//#define CUDA_LIB_EXPORT
//#endif

#define CUDA_LIB_IMPORT
#define CUDA_LIB_EXPORT

#ifndef CUDA_DISABLE_ERROR
#define CUDA_DISABLE_ERROR() std::cerr << __FILE__ << ":line " << __LINE__ << ". Cannot call CUDA function!" << std::endl
#endif // CUDA_DISABLE_ERROR
