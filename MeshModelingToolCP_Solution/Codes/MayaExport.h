#pragma once

#undef MLL_IMPORT
#undef MLL_EXPORT

#if defined _WIN32 || defined _WIN64
#define MLL_IMPORT __declspec(dllimport)
#elif defined __linux__
#define MLL_IMPORT __attribute__((visibility("default")))
#else
#define MLL_IMPORT
#endif

#if defined _WIN32 || defined _WIN64
#define MLL_EXPORT __declspec(dllexport)
#else
#endif

#ifndef MLL_EXPORT
#define MLL_EXPORT
#endif