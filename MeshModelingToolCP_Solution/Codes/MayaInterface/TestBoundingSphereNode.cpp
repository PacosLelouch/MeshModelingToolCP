#include "pch.h"
#include "TestBoundingSphereNode.h"
#include <maya/MItGeometry.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFloatPointArray.h>
#include <maya/MPointArray.h>
#include <maya/MDagModifier.h>

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

MStatus MTestBoundingSphereNode::compute(const MPlug& plug, MDataBlock& block)
{
    return MPxDeformerNode::compute(plug, block);

    //if (!m_geometrySolverShPtr)
    //{
    //    m_geometrySolverShPtr.reset(new MyGeometrySolver3D);
    //}

    //MStatus status = MStatus::kSuccess;

    //MDataHandle hNumIter = block.inputValue(aNumIter, &status);
    //CHECK_MSTATUS_AND_RETURN_IT(status);
    //int numIter = hNumIter.asInt();

    //MDataHandle hSphereProjectionWeight = block.inputValue(aSphereProjectionWeight, &status);
    //CHECK_MSTATUS_AND_RETURN_IT(status);
    //double sphereProjectionWeight = hSphereProjectionWeight.asDouble();

    //MDataHandle hFairnessWeight = block.inputValue(aFairnessWeight, &status);
    //CHECK_MSTATUS_AND_RETURN_IT(status);
    //double fairnessWeight = hFairnessWeight.asDouble();

    //MObject outputMeshObj = getMeshObjectFromOutput(block, multiIndex, &status);
    //CHECK_MSTATUS_AND_RETURN_IT(status);

    //MObject inputMeshObj = getMeshObjectFromInput(block, multiIndex, &status);
    //CHECK_MSTATUS_AND_RETURN_IT(status);

    //m_meshConverterShPtr.reset(new AAShapeUp::MayaToEigenConverter(inputMeshObj));

    //if (!m_meshConverterShPtr->generateEigenMatrices())
    //{
    //    MGlobal::displayError("Fail to generate eigen matrices [input]!");
    //    return MStatus::kFailure;
    //}

    //m_operationShPtr.reset(new AAShapeUp::TestBoundingSphereOperation(m_geometrySolverShPtr));

    //m_operationShPtr->m_sphereProjectionWeight = sphereProjectionWeight;
    //m_operationShPtr->m_LaplacianWeight = fairnessWeight;

    //if (!m_operationShPtr->initialize(m_meshConverterShPtr->getEigenMesh(), {}))
    //{
    //    MGlobal::displayError("Fail to initialize!");
    //    return MStatus::kFailure;
    //}

    //if (!m_operationShPtr->solve(m_meshConverterShPtr->getEigenMesh().m_positions, numIter))
    //{
    //    MGlobal::displayError("Fail to solve!");
    //    return MStatus::kFailure;
    //}

    //auto& finalPositions = m_meshConverterShPtr->getEigenMesh().m_positions;
    //MFloatPointArray finalPositionsMaya;
    //finalPositionsMaya.setLength(finalPositions.cols());
    //for (AAShapeUp::i64 i = 0; i < finalPositions.cols(); ++i)
    //{
    //    finalPositionsMaya[i] = AAShapeUp::fromEigenVec3<MFloatPoint>(finalPositions.col(i));
    //}

    //MFnMesh outFnMesh(outputMeshObj);
    //status = outFnMesh.copyInPlace(inputMeshObj);
    //CHECK_MSTATUS_AND_RETURN_IT(status);
    //status = outFnMesh.setPoints(finalPositionsMaya);
    //CHECK_MSTATUS_AND_RETURN_IT(status);

    //return MStatus::kSuccess;
}

MObject& MTestBoundingSphereNode::accessoryAttribute() const
{
    // TODO: insert return statement here
    return Super::accessoryAttribute();
}

MStatus MTestBoundingSphereNode::accessoryNodeSetup(MDagModifier& cmd)
{
    return Super::accessoryNodeSetup(cmd);
}

MStatus MTestBoundingSphereNode::deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex)
{
    //return Super::deform(block, iter, mat, multiIndex);

    if (!m_geometrySolverShPtr)
    {
        m_geometrySolverShPtr.reset(new MyGeometrySolver3D);
    }

    MStatus status = MStatus::kSuccess;

    MDataHandle hEnvelope = block.inputValue(envelope, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    float envelopeValue = hEnvelope.asFloat();

    MDataHandle hNumIter = block.inputValue(aNumIter, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    int numIter = hNumIter.asInt();

    MDataHandle hSphereProjectionWeight = block.inputValue(aSphereProjectionWeight, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    double sphereProjectionWeight = hSphereProjectionWeight.asDouble();

    MDataHandle hFairnessWeight = block.inputValue(aFairnessWeight, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    double fairnessWeight = hFairnessWeight.asDouble();

    MObject inputMeshObj = getMeshObjectFromInput(block, multiIndex, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // Start check cache.
    InputChangedFlag inputChangedFlag = InputChangedFlag::None;
    if (m_cache.numIter != numIter ||
        m_cache.fairnessWeight != fairnessWeight ||
        m_cache.sphereProjectionWeight != sphereProjectionWeight)
    {
        inputChangedFlag |= InputChangedFlag::Parameter;
        MGlobal::displayInfo("[TestBoundingSphereNode] Change [parameter].");
    }
    //if (m_cache.inputMeshObj != inputMeshObj)
    if (m_cache.inputMeshObj.isNull() || m_cache.inputMeshObj.hasFn(MFn::kMesh))
    {
        inputChangedFlag |= InputChangedFlag::InputMesh;
        MGlobal::displayInfo("[TestBoundingSphereNode] Change [input mesh].");
    }
    m_cache.numIter = numIter;
    m_cache.fairnessWeight = fairnessWeight;
    m_cache.sphereProjectionWeight = sphereProjectionWeight;
    m_cache.inputMeshObj = inputMeshObj;
    // End check cache.

    if (inputChangedFlag != InputChangedFlag::None)
    {
        if ((inputChangedFlag & InputChangedFlag::InputMesh) != InputChangedFlag::None)
        {
            m_meshConverterShPtr.reset(new AAShapeUp::MayaToEigenConverter(inputMeshObj));

            if (!m_meshConverterShPtr->generateEigenMatrices())
            {
                MGlobal::displayError("Fail to generate eigen matrices [input]!");
                return MStatus::kFailure;
            }
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
    }

    // Start write output.
    MPointArray startPositionsMaya, finalPositionsMaya;
    auto& finalPositions = m_meshConverterShPtr->getEigenMesh().m_positions;
    status = iter.allPositions(startPositionsMaya);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    finalPositionsMaya.setLength(finalPositions.cols());
    for (AAShapeUp::i64 i = 0; i < finalPositions.cols(); ++i)
    {
        finalPositionsMaya[i] = envelopeValue * AAShapeUp::fromEigenVec3<MVector>(finalPositions.col(i)) + (1.0f - envelopeValue) * MVector(startPositionsMaya[i]);
    }

    status = iter.setAllPositions(finalPositionsMaya);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    return MStatus::kSuccess;
}
