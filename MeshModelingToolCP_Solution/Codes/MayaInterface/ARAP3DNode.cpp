#include "pch.h"
#include "ARAP3DNode.h"
#include <maya/MItGeometry.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MPointArray.h>

const MTypeId MARAP3DNode::id = 0x01000003;
const MString MARAP3DNode::nodeName = "ARAP3DNode";

MObject MARAP3DNode::aNumIter;

void* MARAP3DNode::creator()
{
    return new MARAP3DNode;
}

MStatus MARAP3DNode::initialize()
{
    MStatus status = MStatus::kSuccess;

    MFnNumericAttribute nAttr;
    MFnTypedAttribute tAttr;

    aNumIter = nAttr.create("numIteration", "niter", MFnNumericData::kInt, 50, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0);
    status = addAttribute(aNumIter);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    status = MGlobal::executeCommand("makePaintable -attrType multiFloat -sm deformer " + MARAP3DNode::nodeName + " weights;");
    CHECK_MSTATUS_AND_RETURN_IT(status);

    status = attributeAffects(aNumIter, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    return MStatus::kSuccess;
}

MStatus MARAP3DNode::deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex)
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

    MObject inputMeshObj = getMeshObjectFromInput(block, multiIndex, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // Start check cache.
    InputChangedFlag inputChangedFlag = InputChangedFlag::None;
    if (m_cache.numIter != numIter)
    {
        inputChangedFlag |= InputChangedFlag::Parameter;
        MGlobal::displayInfo("[ARAP3DNode] Change [parameter].");
    }
    if (m_cache.inputMeshObj.isNull() || !m_cache.inputMeshObj.hasFn(MFn::kMesh))
    {
        inputChangedFlag |= InputChangedFlag::InputMesh;
        MGlobal::displayInfo("[ARAP3DNode] Change [input mesh].");
    }
    m_cache.numIter = numIter;
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

        //// TODO
        //m_operationShPtr.reset(new AAShapeUp::PlanarizationOperation(m_geometrySolverShPtr)); 

        //m_operationShPtr->refMesh = m_meshConverterReferenceShPtr->getEigenMesh();

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

    return MStatus();
}
