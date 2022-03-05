#include "ExportCommon.h"

#if defined _WIN32 || defined _WIN64
    #include <Windows.h>
#else
    #include <stdio.h>
#endif

#include "CUDATest.h"

//MAYA_EXPORT void ExampleFunction()
//{
//    CUDATest::CUDATest::CUDATestFunction();
////#if defined _WIN32 || defined _WIN64
////	MessageBox(NULL, TEXT("Loaded ExampleLibrary.dll from Third Party Plugin sample 111."), TEXT("Third Party Plugin"), MB_OK);
////#else
////    printf("Loaded ExampleLibrary from Third Party Plugin sample");
////#endif
//}
#include "HelloMayaCUDA.h"
#include <maya/MFnPlugin.h>
#include <string>

#if _COMPUTE_USING_CUDA

#endif // _COMPUTE_USING_CUDA

// Maya Plugin creator function
void *helloMaya::creator()
{
    return new helloMaya;
}
// Plugin doIt function
MStatus helloMaya::doIt(const MArgList& argList)
{
    MStatus status;
    MGlobal::displayInfo("Hello World!");
    // <<<your code goes here>>>
    std::string cudaStatusStr;
    CUDATest::CUDATest::CUDATestFunction(true, &cudaStatusStr);
    MString cudaStatusStrM = cudaStatusStr.c_str();
    MGlobal::displayInfo(cudaStatusStrM);

    size_t start_pos = 0;
    while((start_pos = cudaStatusStr.find("\n", start_pos)) != std::string::npos) {
        cudaStatusStr.replace(start_pos, sizeof("\n") - 1, "\\n");
        start_pos += sizeof("\\n") - 1;
    }

    MString command = "confirmDialog -title \"Hello Maya\" -message \"^1s\" -button \"OK\" -defaultButton \"OK\" -messageAlign \"center\"";
    command.format(command, MString(cudaStatusStr.c_str()));
    MGlobal::executeCommand(command);

    return status;
}
// Initialize Maya Plugin upon loading
MAYA_EXPORT MStatus initializePlugin(MObject obj)
{
    MStatus status;
    MFnPlugin plugin( obj, "PacosLelouch", "1.0", "Any");
    status = plugin.registerCommand("helloMayaCUDA", helloMaya::creator );
    if (!status)
        status.perror( "registerCommand failed" );
    return status;
}
// Cleanup Plugin upon unloading
MAYA_EXPORT MStatus uninitializePlugin(MObject obj)
{
    MStatus status;
    MFnPlugin plugin(obj);
    status = plugin.deregisterCommand("helloMayaCUDA");
    if(!status)
        status.perror( "deregisterCommand failed" );
    return status;
}