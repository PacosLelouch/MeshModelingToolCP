#pragma once

#undef DLL_IMPORT
#undef DLL_EXPORT

#if defined _WIN32 || defined _WIN64
#define DLL_IMPORT __declspec(dllimport)
#elif defined __linux__
#define DLL_IMPORT __attribute__((visibility("default")))
#else
#define DLL_IMPORT
#endif

#if defined _WIN32 || defined _WIN64
#define DLL_EXPORT __declspec(dllexport)
#else
#endif

#ifndef DLL_EXPORT
#define DLL_EXPORT
#endif

#ifndef CUDA_DISABLE_ERROR
#define CUDA_DISABLE_ERROR() std::cerr << __FILE__ << ":line " << __LINE__ << ". Cannot call CUDA function!" << std::endl
#endif // CUDA_DISABLE_ERROR
