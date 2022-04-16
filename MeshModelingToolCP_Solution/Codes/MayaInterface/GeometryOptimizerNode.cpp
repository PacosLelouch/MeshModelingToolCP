#include "pch.h"
#include "GeometryOptimizerNode.h"
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MArrayDataBuilder.h>
#include <maya/MFloatArray.h>
#include <maya/MFloatVectorArray.h>
#include <maya/MFloatPointArray.h>

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

bool MGeometryOptimizerNode::isMeshVertexDirty(const MObject& meshToAssign, const MObject& meshInput, AAShapeUp::MeshDirtyFlag checkFlag, bool quiet)
{
    if ((meshToAssign.isNull()) || !meshToAssign.hasFn(MFn::kMesh))
    {
        return !((meshInput.isNull()) || !meshInput.hasFn(MFn::kMesh));
    }
    if ((meshInput.isNull()) || !meshInput.hasFn(MFn::kMesh))
    {
        return !((meshToAssign.isNull()) || !meshToAssign.hasFn(MFn::kMesh));
    }
    char outputBuffer[2048]{ 0 };
    float tolerance = 1e-5f;

    MStatus status = MStatus::kSuccess;
    MFnMesh fnMeshToAssign(meshToAssign);
    MFnMesh fnMeshInput(meshInput);
    // If a non-api operation happens that many have changed the underlying Maya object wrapped by this api object, make sure that the api object references a valid maya object.
    // In particular this call should be used if you are calling mel commands from your plugin. Note that this only applies for mesh shapes: in a plugin node where the dataMesh is being accessed directly this is not necessary.
    // So is it necessary?
    status = fnMeshToAssign.syncObject();
    CHECK_MSTATUS_AND_RETURN(status, true);
    status = fnMeshInput.syncObject();
    CHECK_MSTATUS_AND_RETURN(status, true);

    if ((checkFlag | AAShapeUp::MeshDirtyFlag::PositionDirty) != AAShapeUp::MeshDirtyFlag::None)
    {
        MFloatPointArray positionsToAssign;
        status = fnMeshToAssign.getPoints(positionsToAssign);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        unsigned int positionsToAssignLength = positionsToAssign.length();

        MFloatPointArray positionsInput;
        status = fnMeshInput.getPoints(positionsInput);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        unsigned int positionsInputLength = positionsInput.length();

        if (positionsToAssignLength != positionsInputLength)
        {
            if (!quiet)
            {
                sprintf_s(outputBuffer, "Input mesh position length changed! %d != %d", positionsToAssignLength, positionsInputLength);
                MGlobal::displayInfo(outputBuffer);
            }
            return true;
        }
        for (unsigned int i = 0; i < positionsToAssignLength; ++i)
        {
            if (!positionsToAssign[i].isEquivalent(positionsInput[i], tolerance))
            {
                if (!quiet)
                {
                    sprintf_s(outputBuffer, "Input mesh position value changed! [%d]<%.6f, %.6f, %.6f> != [%d]<%.6f, %.6f, %.6f>",
                        i, positionsToAssign[i].x, positionsToAssign[i].y, positionsToAssign[i].z,
                        i, positionsInput[i].x, positionsInput[i].y, positionsInput[i].z);
                    MGlobal::displayInfo(outputBuffer);
                }
                return true;
            }
        }
    }

    if ((checkFlag | AAShapeUp::MeshDirtyFlag::NormalDirty) != AAShapeUp::MeshDirtyFlag::None)
    {
        MFloatVectorArray normalsToAssign;
        status = fnMeshToAssign.getVertexNormals(true, normalsToAssign);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        unsigned int normalsToAssignLength = normalsToAssign.length();

        MFloatVectorArray normalsInput;
        status = fnMeshInput.getVertexNormals(true, normalsInput);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        unsigned int normalsInputLength = normalsInput.length();

        if (normalsToAssignLength != normalsInputLength)
        {
            if (!quiet)
            {
                sprintf_s(outputBuffer, "Input mesh normal length changed! %d != %d", normalsToAssignLength, normalsInputLength);
                MGlobal::displayInfo(outputBuffer);
            }
            return true;
        }
        for (unsigned int i = 0; i < normalsToAssignLength; ++i)
        {
            if (!normalsToAssign[i].isEquivalent(normalsInput[i], tolerance))
            {
                if (!quiet)
                {
                    sprintf_s(outputBuffer, "Input mesh normal value changed! [%d]<%.6f, %.6f, %.6f> != [%d]<%.6f, %.6f, %.6f>",
                        i, normalsToAssign[i].x, normalsToAssign[i].y, normalsToAssign[i].z,
                        i, normalsInput[i].x, normalsInput[i].y, normalsInput[i].z);
                    MGlobal::displayInfo(outputBuffer);
                }
                return true;
            }
        }
    }

    if ((checkFlag | AAShapeUp::MeshDirtyFlag::ColorDirty) != AAShapeUp::MeshDirtyFlag::None)
    {
        MStringArray colorSetNamesToAssign;
        status = fnMeshToAssign.getColorSetNames(colorSetNamesToAssign);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        unsigned int colorSetNamesToAssignLength = colorSetNamesToAssign.length();

        MStringArray colorSetNamesInput;
        status = fnMeshInput.getColorSetNames(colorSetNamesInput);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        unsigned int colorSetNamesInputLength = colorSetNamesInput.length();

        if (colorSetNamesToAssignLength != colorSetNamesInputLength)
        {
            if (!quiet)
            {
                sprintf_s(outputBuffer, "Input mesh color set length changed! %d != %d", colorSetNamesToAssignLength, colorSetNamesInputLength);
                MGlobal::displayInfo(outputBuffer);
            }
            return true;
        }

        for (unsigned int s = 0; s < colorSetNamesToAssignLength; ++s)
        {
            if (colorSetNamesToAssign[s] != colorSetNamesInput[s])
            {
                if (!quiet)
                {
                    sprintf_s(outputBuffer, "Input mesh color set name changed! %s != %s", colorSetNamesToAssign[s].asChar(), colorSetNamesInput[s].asChar());
                    MGlobal::displayInfo(outputBuffer);
                }
                return true;
            }

            MColorArray colorsToAssign;
            status = fnMeshToAssign.getColors(colorsToAssign, &colorSetNamesToAssign[s]);
            CHECK_MSTATUS_AND_RETURN_IT(status);
            unsigned int colorsToAssignLength = colorsToAssign.length();

            MColorArray colorsInput;
            status = fnMeshInput.getColors(colorsInput, &colorSetNamesInput[s]);
            CHECK_MSTATUS_AND_RETURN_IT(status);
            unsigned int colorsInputLength = colorsInput.length();

            if (colorsToAssignLength != colorsInputLength)
            {
                if (!quiet)
                {
                    sprintf_s(outputBuffer, "Input mesh color length changed! [%d]%d != [%d]%d", s, colorsToAssignLength, s, colorsInputLength);
                    MGlobal::displayInfo(outputBuffer);
                }
                return true;
            }
            for (unsigned int i = 0; i < colorsToAssignLength; ++i)
            {
                if (colorsToAssign[i] != colorsInput[i])
                {
                    if (!quiet)
                    {
                        sprintf_s(outputBuffer, "Input mesh color value changed! [%d, %d]<%.6f, %.6f, %.6f> != [%d, %d]<%.6f, %.6f, %.6f>",
                            s, i, colorsToAssign[i].r, colorsToAssign[i].g, colorsToAssign[i].b,
                            s, i, colorsInput[i].r, colorsInput[i].g, colorsInput[i].b);
                        MGlobal::displayInfo(outputBuffer);
                    }
                    return true;
                }
            }
        }
    }

    if ((checkFlag | AAShapeUp::MeshDirtyFlag::TexCoordsDirty) != AAShapeUp::MeshDirtyFlag::None)
    {
        MStringArray uvSetNamesToAssign;
        status = fnMeshToAssign.getUVSetNames(uvSetNamesToAssign);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        unsigned int uvNamesToAssignLength = uvSetNamesToAssign.length();

        MStringArray uvSetNamesInput;
        status = fnMeshInput.getUVSetNames(uvSetNamesInput);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        unsigned int uvSetNamesInputLength = uvSetNamesInput.length();

        if (uvNamesToAssignLength != uvSetNamesInputLength)
        {
            if (!quiet)
            {
                sprintf_s(outputBuffer, "Input mesh uv set length changed! %d != %d", uvNamesToAssignLength, uvSetNamesInputLength);
                MGlobal::displayInfo(outputBuffer);
            }
            return true;
        }

        for (unsigned int s = 0; s < uvNamesToAssignLength; ++s)
        {
            if (uvSetNamesToAssign[s] != uvSetNamesInput[s])
            {
                if (!quiet)
                {
                    sprintf_s(outputBuffer, "Input mesh uv set name changed! %s != %s", uvSetNamesToAssign[s].asChar(), uvSetNamesInput[s].asChar());
                    MGlobal::displayInfo(outputBuffer);
                }
                return true;
            }

            MFloatArray usToAssign, vsToAssign;
            status = fnMeshToAssign.getUVs(usToAssign, vsToAssign, &uvSetNamesToAssign[s]);
            CHECK_MSTATUS_AND_RETURN_IT(status);
            unsigned int usToAssignLength = usToAssign.length();

            MFloatArray usInput, vsInput;
            status = fnMeshInput.getUVs(usInput, vsInput, &uvSetNamesInput[s]);
            CHECK_MSTATUS_AND_RETURN_IT(status);
            unsigned int usInputLength = usInput.length();

            if (usToAssignLength != usInputLength)
            {
                if (!quiet)
                {
                    sprintf_s(outputBuffer, "Input mesh uv length changed! [%d]%d != [%d]%d", s, usToAssignLength, s, usInputLength);
                    MGlobal::displayInfo(outputBuffer);
                }
                return true;
            }
            for (unsigned int i = 0; i < usToAssignLength; ++i)
            {
                if (usToAssign[i] != usInput[i] || vsToAssign[i] != vsInput[i])
                {
                    if (!quiet)
                    {
                        sprintf_s(outputBuffer, "Input mesh uv value changed! [%d, %d]<%.6f, %.6f> != [%d, %d]<%.6f, %.6f>",
                            s, i, usToAssign[i], vsToAssign[i],
                            s, i, usInput[i], vsInput[i]);
                        MGlobal::displayInfo(outputBuffer);
                    }
                    return true;
                }
            }
        }
    }

    return false;
}

bool MGeometryOptimizerNode::isMeshNotEqualNaive(const MObject& meshToCompare, const MObject& meshInput)
{
    return meshToCompare != meshInput;
}

void MGeometryOptimizerNode::postConstructor()
{
    MPxDeformerNode::postConstructor();
    m_geometrySolverShPtr.reset(new MyGeometrySolver3D);
}

MObject MGeometryOptimizerNode::getMeshObjectFromInputWithoutEval(MDataBlock& block, unsigned int index, MStatus* statusPtr)
{
    MStatus status = MStatus::kSuccess;
    // Get the geometry.
    MArrayDataHandle hInput = block.outputArrayValue(input, &status);
    CHECK_MSTATUS_ASSIGN_AND_RETURN(status, statusPtr, MObject::kNullObj);
    jumpToElement(hInput, index);
    MDataHandle hInputGeom = hInput.outputValue().child(inputGeom);
    MObject inMesh = hInputGeom.asMesh();

    return inMesh;
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
    MArrayDataHandle hOutput = block.outputArrayValue(outputGeom, &status);
    CHECK_MSTATUS_ASSIGN_AND_RETURN(status, statusPtr, MObject::kNullObj);
    jumpToElement(hOutput, index);
    MDataHandle hOutputGeom = hOutput.outputValue();
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
