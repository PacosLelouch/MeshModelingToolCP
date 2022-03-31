#include "pch.h"
#include "TestBoundingSphereNode.h"
#include <maya/MItGeometry.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MPointArray.h>

const MTypeId MTestBoundingSphereNode::id = 0x01000004;
const MString MTestBoundingSphereNode::nodeName = "testBoundingSphereNode";

MObject MTestBoundingSphereNode::aNumIter;
MObject MTestBoundingSphereNode::aSphereProjectionWeight;
MObject MTestBoundingSphereNode::aFairnessWeight;

void* MTestBoundingSphereNode::creator()
{
    return new MTestBoundingSphereNode;
}

MStatus MTestBoundingSphereNode::initialize()
{
    MStatus status = MStatus::kSuccess;

    MFnNumericAttribute nAttr;

    aNumIter = nAttr.create("numIteration", "niter", MFnNumericData::kInt, 50, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0);
    status = addAttribute(aNumIter);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aSphereProjectionWeight = nAttr.create("sphereProjectionWeight", "wsp", MFnNumericData::kDouble, 1.0, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0.0);
    status = addAttribute(aSphereProjectionWeight);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aFairnessWeight = nAttr.create("fairnessWeight", "wfr", MFnNumericData::kDouble, 0.1, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0.0);
    status = addAttribute(aFairnessWeight);
    CHECK_MSTATUS_AND_RETURN_IT(status);


    status = attributeAffects(aNumIter, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aSphereProjectionWeight, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aFairnessWeight, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    return MStatus::kSuccess;
}

MStatus MTestBoundingSphereNode::deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex)
{
    if (!m_geometrySolverShPtr)
    {
        m_geometrySolverShPtr.reset(new MyGeometrySolver3D);
    }

    MStatus status = MStatus::kSuccess;

    MDataHandle hNumIter = block.inputValue(aNumIter, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    int numIter = hNumIter.asInt();

    MDataHandle hSphereProjectionWeight = block.inputValue(aSphereProjectionWeight, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    double sphereProjectionWeight = hSphereProjectionWeight.asDouble();

    MDataHandle hFairnessWeight = block.inputValue(aFairnessWeight, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    double fairnessWeight = hFairnessWeight.asDouble();

    MObject inputMeshObj = getMeshObjectFromInput(block);

    m_meshConverterShPtr.reset(new AAShapeUp::MayaToEigenConverter(inputMeshObj));

    if (m_meshConverterShPtr->generateEigenMatrices())
    {
        MGlobal::displayError("Fail to generate eigen matrices [input]!");
        return MStatus::kFailure;
    }

    m_operationShPtr.reset(new AAShapeUp::TestBoundingSphereOperation(m_geometrySolverShPtr));

    m_operationShPtr->m_sphereProjectionWeight = sphereProjectionWeight;
    m_operationShPtr->m_LaplacianWeight = fairnessWeight;

    if (!m_operationShPtr->initialize(m_meshConverterShPtr->getEigenMesh(), {}))
    {
        MGlobal::displayError("Fail to initialize!");
        return MStatus::kFailure;
    }

    if (!m_operationShPtr->solve(m_meshConverterShPtr->getEigenMesh().m_positions, numIter))
    {
        MGlobal::displayError("Fail to solve!");
        return MStatus::kFailure;
    }

    auto& finalPositions = m_meshConverterShPtr->getEigenMesh().m_positions;
    MPointArray finalPositionsMaya;
    for (AAShapeUp::i64 i = 0; i < finalPositions.cols(); ++i)
    {
        finalPositionsMaya.append(AAShapeUp::fromEigenVec3<MPoint>(finalPositions.col(i)));
    }

    status = iter.setAllPositions(finalPositionsMaya);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    return MStatus::kSuccess;
}
