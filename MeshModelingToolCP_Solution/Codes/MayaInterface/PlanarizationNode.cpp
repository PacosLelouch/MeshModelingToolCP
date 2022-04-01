#include "pch.h"
#include "PlanarizationNode.h"
#include <maya/MItGeometry.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MPointArray.h>

const MTypeId MPlanarizationNode::id = 0x01000001;
const MString MPlanarizationNode::nodeName = "planarizationNode";

MObject MPlanarizationNode::aNumIter;
MObject MPlanarizationNode::aPlanarityWeight;
MObject MPlanarizationNode::aClosenessWeight;
MObject MPlanarizationNode::aFairnessWeight;
MObject MPlanarizationNode::aRelativeFairnessWeight;
MObject MPlanarizationNode::aReferenceMesh;

void* MPlanarizationNode::creator()
{
    return new MPlanarizationNode;
}

MStatus MPlanarizationNode::initialize()
{
    //MStatus status = Super::initialize();
    //CHECK_MSTATUS_AND_RETURN_IT(status);
    MStatus status = MStatus::kSuccess;

    MFnNumericAttribute nAttr;
    MFnTypedAttribute tAttr;

    aNumIter = nAttr.create("numIteration", "niter", MFnNumericData::kInt, 50, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0);
    status = addAttribute(aNumIter);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aPlanarityWeight = nAttr.create("planarityWeight", "wpl", MFnNumericData::kDouble, 150.0, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0.0);
    status = addAttribute(aPlanarityWeight);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aClosenessWeight = nAttr.create("closenessWeight", "wcl", MFnNumericData::kDouble, 10.0, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0.0);
    status = addAttribute(aClosenessWeight);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aFairnessWeight = nAttr.create("fairnessWeight", "wfr", MFnNumericData::kDouble, 0.1, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0.0);
    status = addAttribute(aFairnessWeight);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aRelativeFairnessWeight = nAttr.create("relativeFairnessWeight", "wrfr", MFnNumericData::kDouble, 1.0, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0.0);
    status = addAttribute(aRelativeFairnessWeight);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aReferenceMesh = tAttr.create("referenceMesh", "rfmesh", MFnData::kMesh, MObject::kNullObj, &status);
    MAYA_ATTR_INPUT(nAttr);
    status = addAttribute(aReferenceMesh);
    CHECK_MSTATUS_AND_RETURN_IT(status);


    status = attributeAffects(aNumIter, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aPlanarityWeight, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aClosenessWeight, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aFairnessWeight, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aRelativeFairnessWeight, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aReferenceMesh, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    return MStatus::kSuccess;
}

MStatus MPlanarizationNode::deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex)
{
    if (!m_geometrySolverShPtr)
    {
        m_geometrySolverShPtr.reset(new MyGeometrySolver3D);
    }

    MStatus status = MStatus::kSuccess;

    MDataHandle hNumIter = block.inputValue(aNumIter, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    int numIter = hNumIter.asInt();

    MDataHandle hClosenessWeight = block.inputValue(aClosenessWeight, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    double closenessWeight = hClosenessWeight.asDouble();

    MDataHandle hPlanarityWeight = block.inputValue(aPlanarityWeight, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    double planarityWeight = hPlanarityWeight.asDouble();

    MDataHandle hFairnessWeight = block.inputValue(aFairnessWeight, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    double fairnessWeight = hFairnessWeight.asDouble();

    MDataHandle hRelativeFairnessWeight = block.inputValue(aRelativeFairnessWeight, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    double relativeFairnessWeight = hRelativeFairnessWeight.asDouble();

    MDataHandle hRefMeshData = block.inputValue(aReferenceMesh, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MObject inputMeshObj = getMeshObjectFromInput(block, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MObject refMeshObj = hRefMeshData.asMesh();
    bool referenceSameAsInput = refMeshObj.isNull();
    if (referenceSameAsInput)
    {
        refMeshObj = getMeshObjectFromInput(block, &status);
    }
    
    m_meshConverterShPtr.reset(new AAShapeUp::MayaToEigenConverter(inputMeshObj));
    m_meshConverterReferenceShPtr.reset(new AAShapeUp::MayaToEigenConverter(refMeshObj));

    if (!m_meshConverterShPtr->generateEigenMatrices())
    {
        MGlobal::displayError("Fail to generate eigen matrices [input]!");
        return MStatus::kFailure;
    }
    if (!referenceSameAsInput && m_meshConverterReferenceShPtr->generateEigenMatrices())
    {
        MGlobal::displayError("Fail to generate eigen matrices [reference]!");
        return MStatus::kFailure;
    }
    
    m_operationShPtr.reset(new AAShapeUp::PlanarizationOperation(m_geometrySolverShPtr));

    m_operationShPtr->refMesh = m_meshConverterReferenceShPtr->getEigenMesh();
    m_operationShPtr->closeness_weight = closenessWeight;
    m_operationShPtr->planarity_weight = planarityWeight;
    m_operationShPtr->laplacian_weight = fairnessWeight;
    m_operationShPtr->relative_laplacian_weight = relativeFairnessWeight;

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

    //AAShapeUp::MeshDirtyFlag normalDirtyFlag = AAShapeUp::regenerateNormals(m_meshConverterShPtr->getEigenMesh());

    //if (!m_meshConverterShPtr->updateTargetMesh(m_operationShPtr->getMeshDirtyFlag() | normalDirtyFlag, ?, true))
    //{
    //    std::cout << "Fail to update source mesh!" << std::endl;
    //    return;
    //}

    return MStatus::kSuccess;
}
