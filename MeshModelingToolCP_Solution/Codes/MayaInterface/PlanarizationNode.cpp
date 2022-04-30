#include "pch.h"
#include "PlanarizationNode.h"
#include <maya/MItGeometry.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MPointArray.h>

const MTypeId MPlanarizationNode::id = 0x01000001;
const MString MPlanarizationNode::nodeName = "planarizationNode";

MObject MPlanarizationNode::aNumIter;
MObject MPlanarizationNode::aMaxDisplacementVisualization;
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

    aNumIter = nAttr.create("numIteration", "nIter", MFnNumericData::kInt, 50, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0);
    status = addAttribute(aNumIter);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aMaxDisplacementVisualization = nAttr.create("maxDisplacementVisualization", "maxDispVis", MFnNumericData::kDouble, 0.0, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0.0);
    status = addAttribute(aMaxDisplacementVisualization);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aPlanarityWeight = nAttr.create("planarityWeight", "wpl", MFnNumericData::kDouble, 1500.0, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0.0);
    status = addAttribute(aPlanarityWeight);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aClosenessWeight = nAttr.create("closenessWeight", "wcl", MFnNumericData::kDouble, 100.0, &status);
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
    status = attributeAffects(aMaxDisplacementVisualization, outputGeom);
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

    MDataHandle hEnvelope = block.inputValue(envelope, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    float envelopeValue = hEnvelope.asFloat();

    MDataHandle hNumIter = block.inputValue(aNumIter, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    int numIter = hNumIter.asInt();

    MDataHandle hMaxDisplacementVisualization = block.inputValue(aMaxDisplacementVisualization, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    double maxDisplacementVisualization = hMaxDisplacementVisualization.asDouble();

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

    MDataHandle hReferenceMeshData = block.inputValue(aReferenceMesh, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    //MObject inputMeshObj = getMeshObjectFromInputWithoutEval(block, multiIndex, &status);
    MObject inputMeshObj = getMeshObjectFromInput(block, multiIndex, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MObject referenceMeshObj = hReferenceMeshData.asMesh();
    if (referenceMeshObj.isNull())
    {
        return MStatus::kFailure;
    }
    //bool referenceSameAsInput = referenceMeshObj.isNull();
    //if (referenceSameAsInput)
    //{
    //    referenceMeshObj = inputMeshObj;//getMeshObjectFromInputWithoutEval(block, multiIndex, &status);
    //}

    // Start check cache.
    InputChangedFlag inputChangedFlag = InputChangedFlag::None;
    if (m_cache.maxDisplacementVisualization != maxDisplacementVisualization)
    {
        inputChangedFlag |= InputChangedFlag::Visualization;
        MGlobal::displayInfo("[" + nodeName + "] Change [visualization].");
    }
    if (m_cache.numIter != numIter ||
        m_cache.planarityWeight != planarityWeight ||
        m_cache.closenessWeight != closenessWeight ||
        m_cache.fairnessWeight != fairnessWeight ||
        m_cache.relativeFairnessWeight != relativeFairnessWeight)
    {
        inputChangedFlag |= InputChangedFlag::Parameter;
        MGlobal::displayInfo("[" + nodeName + "] Change [parameter].");
    }
    if (isMeshVertexDirty(m_cache.inputMeshObj, inputMeshObj))
    {
        inputChangedFlag |= InputChangedFlag::InputMesh;
        MGlobal::displayInfo("[" + nodeName + "] Change [input mesh].");
    }
    if (m_cache.referenceMeshObj != referenceMeshObj)
    {
        inputChangedFlag |= InputChangedFlag::ReferenceMesh;
        MGlobal::displayInfo("[" + nodeName + "] Change [reference mesh].");
    }
    m_cache.numIter = numIter;
    m_cache.maxDisplacementVisualization = maxDisplacementVisualization;
    m_cache.planarityWeight = planarityWeight;
    m_cache.closenessWeight = closenessWeight;
    m_cache.fairnessWeight = fairnessWeight;
    m_cache.relativeFairnessWeight = relativeFairnessWeight;
    m_cache.inputMeshObj = inputMeshObj;
    m_cache.referenceMeshObj = referenceMeshObj;
    // End check cache.
    
    if (inputChangedFlag > InputChangedFlag::Visualization)
    {
        if ((inputChangedFlag & InputChangedFlag::InputMesh) != InputChangedFlag::None)
        {
            m_meshConverterShPtr.reset(new AAShapeUp::MayaToEigenConverter(inputMeshObj));

            if (!m_meshConverterShPtr->generateEigenMesh())
            {
                MGlobal::displayError("Fail to generate eigen matrices [input]!");
                return MStatus::kFailure;
            }

            m_operationShPtr.reset(new AAShapeUp::PlanarizationOperation(m_geometrySolverShPtr));
        }
        if ((inputChangedFlag & InputChangedFlag::ReferenceMesh) != InputChangedFlag::None)
        {
            m_meshConverterReferenceShPtr.reset(new AAShapeUp::MayaToEigenConverter(referenceMeshObj));

            //if (!referenceSameAsInput && !m_meshConverterReferenceShPtr->generateEigenMatrices())
            if (!m_meshConverterReferenceShPtr->generateEigenMesh())
            {
                MGlobal::displayError("Fail to generate eigen matrices [reference]!");
                return MStatus::kFailure;
            }
        }
        //m_meshConverterShPtr->resetOutputEigenMeshToInitial();

        m_operationShPtr->refMesh = m_meshConverterReferenceShPtr->getOutputEigenMesh();
        m_operationShPtr->closeness_weight = closenessWeight;
        m_operationShPtr->planarity_weight = planarityWeight;
        m_operationShPtr->laplacian_weight = fairnessWeight;
        m_operationShPtr->relative_laplacian_weight = relativeFairnessWeight;

        if (!m_operationShPtr->initialize(m_meshConverterShPtr->getInitialEigenMesh(), {}))
        {
            MGlobal::displayError("Fail to initialize!");
            return MStatus::kFailure;
        }

        if (!m_operationShPtr->solve(m_meshConverterShPtr->getOutputEigenMesh().m_positions, numIter))
        {
            MGlobal::displayError("Fail to solve!");
            return MStatus::kFailure;
        }
    }

    if ((inputChangedFlag & InputChangedFlag::Visualization) != InputChangedFlag::None)
    {
        if (maxDisplacementVisualization != 0.0)
        {
            AAShapeUp::MeshDirtyFlag colorDirtyFlag = m_operationShPtr->visualizeDisplacements(m_meshConverterShPtr->getOutputEigenMesh().m_colors, true);

            MObject outputMeshObj = getMeshObjectFromOutput(block, multiIndex, &status);
            status = m_meshConverterShPtr->updateTargetMesh(colorDirtyFlag, outputMeshObj, false);
            CHECK_MSTATUS_AND_RETURN_IT(status);
        }
    }
    
    // Start write output.
    MPointArray startPositionsMaya, finalPositionsMaya;
    auto& finalPositions = m_meshConverterShPtr->getOutputEigenMesh().m_positions;
    status = iter.allPositions(startPositionsMaya);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    finalPositionsMaya.setLength(finalPositions.cols());
    for (AAShapeUp::i64 i = 0; i < finalPositions.cols(); ++i)
    {
        finalPositionsMaya[i] = envelopeValue * AAShapeUp::fromEigenVec3<MVector>(finalPositions.col(i)) + (1.0f - envelopeValue) * MVector(startPositionsMaya[i]);
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
