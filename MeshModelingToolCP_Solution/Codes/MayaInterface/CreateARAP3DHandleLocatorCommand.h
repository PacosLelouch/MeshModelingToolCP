#pragma once

#include "MayaNodeCommon.h"
#include <maya/MPxCommand.h>
#include <maya/MArgList.h>
#include <maya/MObject.h>
#include <maya/MGlobal.h>
#include <maya/MPxCommand.h>
#include <maya/MStringArray.h>

class MCreateARAP3DHandleLocatorCommand : public MPxCommand
{
public:
    static void *creator();

public:
    virtual MStatus doIt(const MArgList& args);

    static MStatus findDeformerNodeNamesFromSelectedShape(MStringArray& deformerNodeNames, const MString& shapeName, const MString& deformerType = "THdeformer", bool displayExecution = false);

    static MStatus findVerticesFromSelections(MIntArray& vertices, const MStringArray& selections);

public:
    static const MString commandName;
};