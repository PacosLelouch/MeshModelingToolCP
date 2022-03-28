#include "pch.h"
#include "GeometryOptimizerNode.h"
#include "maya/MFnUnitAttribute.h"
#include "maya/MArrayDataBuilder.h"

MObject MGeometryOptimizerNode::aTime;
bool MGeometryOptimizerNode::initialized = false;

MStatus MGeometryOptimizerNode::initialize()
{
    if (initialized)
    {
        return MStatus::kSuccess;
    }

    MStatus status;

    MFnUnitAttribute uAttr;

    aTime = uAttr.create("time", "t", MFnUnitAttribute::kTime, 1.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = addAttribute(aTime);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    status = attributeAffects(aTime, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    initialized = true;
    return MStatus::kSuccess;
}

MStatus MGeometryOptimizerNode::jumpToElement(MArrayDataHandle& hArray, unsigned int index)
{
    // Borrowed from Chad Vernon. Thanks Chad!
    MStatus status = hArray.jumpToElement(index);
    if (MFAIL(status)) {
        MArrayDataBuilder builder = hArray.builder(&status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        builder.addElement(index, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        status = hArray.set(builder);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        status = hArray.jumpToElement(index);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    return status;
}

MFnMesh MGeometryOptimizerNode::getFnMeshFromInput(MDataBlock& block)
{
    // Get the geometry.
    MArrayDataHandle hInput = block.inputArrayValue(input);
    jumpToElement(hInput, 0);
    MDataHandle hInputGeom = hInput.inputValue().child(inputGeom);
    MObject inmesh = hInputGeom.asMesh();
    return MFnMesh(inmesh);
}
