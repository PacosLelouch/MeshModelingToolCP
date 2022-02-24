#pragma once

#undef MAYA_IMPORT
#undef MAYA_EXPORT

#if defined _WIN32 || defined _WIN64
#define MAYA_IMPORT __declspec(dllimport)
#elif defined __linux__
#define MAYA_IMPORT __attribute__((visibility("default")))
#else
#define MAYA_IMPORT
#endif

#if defined _WIN32 || defined _WIN64
#define MAYA_EXPORT __declspec(dllexport)
#else
#endif

#ifndef MAYA_EXPORT
#define MAYA_EXPORT
#endif