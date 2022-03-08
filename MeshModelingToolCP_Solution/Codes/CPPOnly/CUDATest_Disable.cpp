#include "CUDATest.h"
#include <string>
#if defined _WIN32 || defined _WIN64
#include <Windows.h>

#else
#include <stdio.h>
#endif

namespace CUDATest 
{

void CUDATest::CUDATestFunction(bool quiet, std::string* outStr) 
{
    WCHAR message[16384]{ 0 };
    //memset(message, 0, sizeof(message));
    LPWSTR cursor = message;
    cursor += wsprintf(cursor, 
        TEXT("cuda_query() device count = 0 i.e., there is not") 
        TEXT(" a CUDA-supported GPU!!!\n"));
    
    if (!quiet) {
#if defined _WIN32 || defined _WIN64
        MessageBox(NULL, message, TEXT("Third Party Plugin CUDA Test"), MB_OK);
#else
        wprintf(TEXT("%s"), message);
#endif
    }

    if (outStr)
    {
        CHAR messageANSI[16384]{ 0 };
        if (CharToOemW(message, messageANSI))
        {
            outStr->assign(messageANSI);
        }
    }
}
}