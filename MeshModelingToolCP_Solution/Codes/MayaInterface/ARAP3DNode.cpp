#include "pch.h"
#include "ARAP3DNode.h"
#include <maya/MItGeometry.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MPointArray.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MArrayDataBuilder.h>
#include <maya/MPlug.h>

const MTypeId MARAP3DNode::id = 0x01000003;
const MString MARAP3DNode::nodeName = "ARAP3DNode";

MObject MARAP3DNode::aNumIter;
MObject MARAP3DNode::aMaxDisplacementVisualization;
MObject MARAP3DNode::aDeformationWeight;
MObject MARAP3DNode::aHandlePositions;
MObject MARAP3DNode::aHandleIndices;

void* MARAP3DNode::creator()
{
    return new MARAP3DNode;
}

MStatus MARAP3DNode::initialize()
{
    MStatus status = MStatus::kSuccess;

    MFnNumericAttribute nAttr;
    MFnTypedAttribute tAttr;
    MFnMatrixAttribute mAttr;

    aNumIter = nAttr.create("numIteration", "nIter", MFnNumericData::kInt, 20, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0);
    status = addAttribute(aNumIter);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aMaxDisplacementVisualization = nAttr.create("maxDisplacementVisualization", "maxDispVis", MFnNumericData::kDouble, 0.0, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0.0);
    status = addAttribute(aMaxDisplacementVisualization);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aDeformationWeight = nAttr.create("deformationWeight", "wd", MFnNumericData::kDouble, 1000.0, &status);
    MAYA_ATTR_INPUT(nAttr);
    nAttr.setMin(0.0);
    status = addAttribute(aDeformationWeight);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aHandlePositions = nAttr.create("handlePositions", "hdlPts", MFnNumericData::k3Double, 0.0, &status);
    MAYA_ATTR_INPUT(nAttr);
    //nAttr.setWritable(false);
    nAttr.setArray(true);
    //nAttr.setHidden(true);
    nAttr.setDisconnectBehavior(MFnAttribute::kDelete);
    //mAttr.setDefault(MMatrix::identity);
    status = addAttribute(aHandlePositions);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    aHandleIndices = nAttr.create("handleIndices", "hdlIdxs", MFnNumericData::kInt, -1, &status);
    MAYA_ATTR_INPUT(nAttr);
    //nAttr.setWritable(false);
    nAttr.setArray(true);
    //nAttr.setHidden(true);
    nAttr.setDisconnectBehavior(MFnAttribute::kDelete);
    status = addAttribute(aHandleIndices);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    //status = MGlobal::executeCommand("makePaintable -attrType multiFloat -sm deformer " + MARAP3DNode::nodeName + " weights;");
    //CHECK_MSTATUS_AND_RETURN_IT(status);

    status = attributeAffects(aNumIter, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aMaxDisplacementVisualization, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aHandlePositions, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aHandleIndices, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    return MStatus::kSuccess;
}

void MARAP3DNode::postConstructor()
{
    Super::postConstructor();
    MStatus status = MStatus::kSuccess;
    MObject thisNode = thisMObject();
    MPlug plugHandlePositions(thisNode, aHandlePositions);
    MPlug plugHandleIndices(thisNode, aHandleIndices);

    MDataBlock block = forceCache();
    MDataHandle hHandlePositions = plugHandlePositions.constructHandle(block);
    MDataHandle hHandleIndices = plugHandleIndices.constructHandle(block);

    MGlobal::displayInfo(nodeName + plugHandlePositions.name() + " has " + plugHandlePositions.evaluateNumElements() + " elements." + (plugHandlePositions.isArray() ? " IsArray." : " NotArray."));
    MGlobal::displayInfo(nodeName + plugHandleIndices.name() + " has " + plugHandleIndices.evaluateNumElements() + " elements." + (plugHandleIndices.isArray() ? " IsArray." : " NotArray."));

    //MArrayDataHandle ahHandlePositions(hHandlePositions, &status);
    //MArrayDataHandle ahHandleIndices(hHandleIndices, &status);

    //MArrayDataBuilder abHandlePositions = ahHandlePositions.builder(&status);
    //MArrayDataBuilder abHandleIndices = ahHandleIndices.builder(&status);

    //ahHandlePositions.set(abHandlePositions);
    //ahHandleIndices.set(abHandleIndices);

    //plugHandlePositions.setValue(hHandlePositions);
    //plugHandleIndices.setValue(hHandleIndices);
}

MStatus MARAP3DNode::deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex)
{
    if (!m_geometrySolverShPtr)
    {
        m_geometrySolverShPtr.reset(new MyGeometrySolver3D);
    }

    MStatus status = MStatus::kSuccess;
    MMatrix worldToLocal = mat.inverse();

    MDataHandle hEnvelope = block.inputValue(envelope, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    float envelopeValue = hEnvelope.asFloat();

    MDataHandle hNumIter = block.inputValue(aNumIter, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    int numIter = hNumIter.asInt();

    MDataHandle hMaxDisplacementVisualization = block.inputValue(aMaxDisplacementVisualization, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    double maxDisplacementVisualization = hMaxDisplacementVisualization.asDouble();

    MDataHandle hDeformationWeight = block.inputValue(aDeformationWeight, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    double deformationWeight = hDeformationWeight.asDouble();

    MArrayDataHandle hHandlePositions = block.inputArrayValue(aHandlePositions, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    unsigned int numHandlePositions = hHandlePositions.elementCount(&status);
    std::vector<MPoint> handlePositions;
    handlePositions.resize(numHandlePositions);
    for (unsigned int i = 0; i < numHandlePositions; ++i)
    {
        hHandlePositions.jumpToElement(i);
        double3& handlePositionDouble3 = hHandlePositions.inputValue().asDouble3();
        handlePositions[i] = MPoint(handlePositionDouble3[0], handlePositionDouble3[1], handlePositionDouble3[2]);
    }

    MArrayDataHandle hHandleIndices = block.inputArrayValue(aHandleIndices, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    unsigned int numHandleIndices = hHandleIndices.elementCount(&status);
    std::vector<int> handleIndices;
    handleIndices.resize(numHandleIndices);
    for (unsigned int i = 0; i < numHandleIndices; ++i)
    {
        hHandleIndices.jumpToElement(i);
        int index = hHandleIndices.inputValue().asInt();
        handleIndices[i] = index;
    }

    //MObject inputMeshObj = getMeshObjectFromInputWithoutEval(block, multiIndex, &status);
    MObject inputMeshObj = getMeshObjectFromInput(block, multiIndex, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // Start check cache.
    InputChangedFlag inputChangedFlag = InputChangedFlag::None;
    if (m_cache.maxDisplacementVisualization != maxDisplacementVisualization)
    {
        inputChangedFlag |= InputChangedFlag::Visualization;
        MGlobal::displayInfo("[" + nodeName + "] Change [visualization].");
    }
    if (m_cache.numIter != numIter ||
        m_cache.deformationWeight != deformationWeight ||
        m_cache.handleIndices != handleIndices ||
        m_cache.handlePositions != handlePositions)
    {
        inputChangedFlag |= InputChangedFlag::Parameter;
        MGlobal::displayInfo("[" + nodeName + "] Change [parameter].");
    }
    if (isMeshVertexDirty(m_cache.inputMeshObj, inputMeshObj))
    {
        inputChangedFlag |= InputChangedFlag::InputMesh;
        MGlobal::displayInfo("[" + nodeName + "] Change [input mesh].");
    }
    m_cache.numIter = numIter;
    m_cache.maxDisplacementVisualization = maxDisplacementVisualization;
    m_cache.deformationWeight = deformationWeight;
    m_cache.inputMeshObj = inputMeshObj;
    m_cache.handleIndices = handleIndices;
    m_cache.handlePositions = handlePositions;
    // End check cache.


    if (inputChangedFlag > InputChangedFlag::Visualization)
    {
        bool usingTetCache = true;
        if ((inputChangedFlag & InputChangedFlag::InputMesh) != InputChangedFlag::None)
        {
            m_meshConverterShPtr.reset(new AAShapeUp::MayaToEigenConverter(inputMeshObj));

            if (!m_meshConverterShPtr->generateEigenMesh())
            {
                MGlobal::displayError("Fail to generate eigen matrices [input]!");
                return MStatus::kFailure;
            }

            m_operationShPtr.reset(new AAShapeUp::ARAP3DOperation(m_geometrySolverShPtr)); 
            usingTetCache = false;
        }
        //m_meshConverterShPtr->resetOutputEigenMeshToInitial();
        m_meshConverterShPtr->getOutputEigenMesh().m_positions = m_meshConverterShPtr->getInitialEigenMesh().m_positions;
        for (size_t i = 0; i < handleIndices.size(); ++i)
        {
            int idx = handleIndices[i];
            MPoint handlePosLocal = handlePositions[i] * worldToLocal;
            m_meshConverterShPtr->getOutputEigenMesh().m_positions.col(idx) = AAShapeUp::toEigenVec3(handlePosLocal);
            //// Begin TEST
            //std::cout << "[Debug] handles(" << i << ")[" << idx << "] = <" << handlePosLocal.x << ", " << handlePosLocal.y << ", " << handlePosLocal.z << ">" << std::endl;
            //// End TEST
        }
        //// Begin TEST
        //std::cout << "[Debug] Update node " << std::endl;
        //for (Eigen::Index i = 0; i < m_meshConverterShPtr->getOutputEigenMesh().m_positions.cols(); ++i)
        //{
        //    std::cout << "[Debug] points[" << i << "] = <" << m_meshConverterShPtr->getOutputEigenMesh().m_positions(0, i) << ", " << m_meshConverterShPtr->getOutputEigenMesh().m_positions(1, i) << ", " << m_meshConverterShPtr->getOutputEigenMesh().m_positions(2, i) << ">" << std::endl;
        //}
        //// End TEST

        m_operationShPtr->m_deformationWeight = deformationWeight;
        if (usingTetCache)
        {
            m_operationShPtr->markUsingCache();
        }

        if (!m_operationShPtr->initialize(m_meshConverterShPtr->getInitialEigenMesh(), handleIndices, &m_meshConverterShPtr->getOutputEigenMesh().m_positions))
        {
            MGlobal::displayError("Fail to initialize!");
            return MStatus::kFailure;
        }

        numIter = handleIndices.size() == 0 ? 0 : numIter; // No necessary to solve if no handles.

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

    return MStatus();
}
