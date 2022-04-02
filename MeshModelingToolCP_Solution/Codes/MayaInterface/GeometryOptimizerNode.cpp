#include "pch.h"
#include "GeometryOptimizerNode.h"
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MArrayDataBuilder.h>

#define CHECK_MSTATUS_ASSIGN_AND_RETURN(_status,_targetPtr,_retVal) \
    if((_targetPtr)) \
    { \
        *(_targetPtr) = _status; \
    } \
    CHECK_MSTATUS_AND_RETURN(_status, _retVal)

//MObject MGeometryOptimizerNode::aTime;
//MObject MGeometryOptimizerNode::aNumIter;
//bool MGeometryOptimizerNode::initialized = false;
//
//MStatus MGeometryOptimizerNode::initialize()
//{
//    if (initialized)
//    {
//        return MStatus::kSuccess;
//    }
//
//    MStatus status = MStatus::kSuccess;
//
//    MFnNumericAttribute nAttr;
//
//    aNumIter = nAttr.create("numIteration", "niter", MFnNumericData::kInt, 50, &status);
//    MAYA_ATTR_INPUT(nAttr);
//    nAttr.setMin(0);
//    status = addAttribute(aNumIter);
//    CHECK_MSTATUS_AND_RETURN_IT(status);
//
//    status = attributeAffects(aNumIter, outputGeom);
//    CHECK_MSTATUS_AND_RETURN_IT(status);
//
//    //MFnUnitAttribute uAttr;
//
//    //aTime = uAttr.create("time", "t", MFnUnitAttribute::kTime, 1.0, &status);
//    //CHECK_MSTATUS_AND_RETURN_IT(status);
//    //status = addAttribute(aTime);
//    //CHECK_MSTATUS_AND_RETURN_IT(status);
//
//    //status = attributeAffects(aTime, outputGeom);
//    //CHECK_MSTATUS_AND_RETURN_IT(status);
//
//    initialized = true;
//    return MStatus::kSuccess;
//}

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

MObject MGeometryOptimizerNode::getMeshObjectFromInput(MDataBlock& block, unsigned int index, MStatus* statusPtr)
{
    MStatus status = MStatus::kSuccess;
    // Get the geometry.
    MArrayDataHandle hInput = block.inputArrayValue(input, &status);
    CHECK_MSTATUS_ASSIGN_AND_RETURN(status, statusPtr, MObject::kNullObj);
    jumpToElement(hInput, index);
    MDataHandle hInputGeom = hInput.inputValue().child(inputGeom);
    MObject inMesh = hInputGeom.asMesh();

    //MDataHandle hInputGeom = block.inputValue(inputGeom, &status);
    //CHECK_MSTATUS_ASSIGN_AND_RETURN(status, statusPtr, MObject::kNullObj);
    //MObject inMesh = hInputGeom.asMesh();

    //MArrayDataHandle hInputGeoms = block.inputArrayValue(inputGeom, &status);
    //jumpToElement(hInputGeoms, 0);
    //MDataHandle hInputGeom = hInputGeoms.inputValue(&status);
    //MObject inMesh = hInputGeom.asMesh();

    return inMesh;
}

MObject MGeometryOptimizerNode::getMeshObjectFromOutput(MDataBlock& block, unsigned int index, MStatus* statusPtr)
{
    MStatus status = MStatus::kSuccess;
    // Get the geometry.
    MArrayDataHandle hOutput = block.outputArrayValue(input, &status);
    CHECK_MSTATUS_ASSIGN_AND_RETURN(status, statusPtr, MObject::kNullObj);
    jumpToElement(hOutput, index);
    MDataHandle hOutputGeom = hOutput.outputValue().child(inputGeom);
    MObject outMesh = hOutputGeom.asMesh();

    //MDataHandle hOutputGeom = block.outputValue(outputGeom, &status);
    //CHECK_MSTATUS_ASSIGN_AND_RETURN(status, statusPtr, MObject::kNullObj);
    //MObject outMesh = hOutputGeom.asMesh();

    //MArrayDataHandle hOutputGeoms = block.outputArrayValue(inputGeom, &status);
    //jumpToElement(hOutputGeoms, 0);
    //MDataHandle hOutputGeom = hOutputGeoms.outputValue(&status);
    //MObject outMesh = hOutputGeom.asMesh();

    return outMesh;
}
