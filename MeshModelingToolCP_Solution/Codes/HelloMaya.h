#pragma once

#include "CUDAExportCommon.h"

//MAYA_IMPORT void ExampleFunction();

#include <maya/MArgList.h>
#include <maya/MObject.h>
#include <maya/MGlobal.h>
#include <maya/MPxCommand.h>
// custom Maya command
class HelloMaya : public MPxCommand
{
public:
    HelloMaya () {};
    virtual MStatus doIt(const MArgList& args);
    static void *creator();
};