#include "ARAP3DHandleLocator.h"
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MMatrixArray.h>
#include <maya/MIntArray.h>
#include <maya/MSelectionList.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnMatrixData.h>
#include <maya/MPxManipulatorNode.h>
//#define MAYA_ENABLE_VP2_PLUGIN_LOCATOR_LEGACY_DRAW 1
//#define MAYA_VP2_USE_VP1_SELECTION 1

const MTypeId MARAP3DHandleLocatorNode::id = 0x02000001;
const MString MARAP3DHandleLocatorNode::nodeName = "ARAPHandleLocator";

const MString MARAP3DHandleLocatorNode::drawDbClassification = "drawdb/geometry/ARAPHandleLocator";
const MString MARAP3DHandleLocatorNode::drawRegistrantId = "AAShapeUp";

//MObject MARAP3DHandleLocatorNode::aPointSize;
MObject MARAP3DHandleLocatorNode::aVertexIndex;

void* MARAP3DHandleLocatorNode::creator()
{
    return new MARAP3DHandleLocatorNode;
}

MStatus MARAP3DHandleLocatorNode::initialize()
{
    MStatus status = MStatus::kSuccess;


    MFnNumericAttribute nAttr;
    MFnTypedAttribute tAttr;

    //aPointSize = nAttr.create("pointSize", "ps", MFnNumericData::kFloat, 10.0f, &status);
    //MAYA_ATTR_INPUT(nAttr);
    //status = addAttribute(aPointSize);
    //CHECK_MSTATUS_AND_RETURN_IT(status);

    aVertexIndex = nAttr.create("vertexIndex", "vidx", MFnNumericData::kInt, -1, &status);
    MAYA_ATTR_INPUT(nAttr);
    //nAttr.setWritable(false);
    status = addAttribute(aVertexIndex);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    return status;
}

void MARAP3DHandleLocatorNode::postConstructor()
{
    MObject oThis = thisMObject();
    MFnDependencyNode fnNode(oThis);
    fnNode.setName(nodeName + "Shape#");
}

bool MARAP3DHandleLocatorNode::useClosestPointForSelection() const
{
    return false;
}

//bool MARAP3DHandleLocatorNode::isBounded() const
//{
//    return false;
//}
//
//bool MARAP3DHandleLocatorNode::isTransparent() const
//{
//    return true;
//}
//
//void MARAP3DHandleLocatorNode::draw(M3dView& view, const MDagPath& path, M3dView::DisplayStyle style, M3dView::DisplayStatus status)
//{
//    MPxLocatorNode::draw(view, path, style, status);
//    return;
//
//    //view.beginGL();
//    //glPushAttrib(GL_CURRENT_BIT);
//    //glEnable(GL_BLEND);
//    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_ALPHA);
//    //glDepthMask(GL_FALSE);
//
//    //MColor solidColor, wireColor;
//    //if (status == M3dView::kActive)
//    //{
//    //    solidColor = MColor(1.0f, 1.0f, 1.0f, 0.1f);
//    //    wireColor = MColor(1.0f, 1.0f, 1.0f, 1.0f);
//    //}
//    //else if (status == M3dView::kLead)
//    //{
//    //    solidColor = MColor(0.26f, 1.0f, 0.64f, 0.1f);
//    //    wireColor = MColor(0.26f, 1.0f, 0.64f, 1.0f);
//    //}
//    //else
//    //{
//    //    solidColor = MColor(1.0f, 1.0f, 0.0f, 0.1f);
//    //    wireColor = MColor(1.0f, 1.0f, 0.0f, 1.0f);
//    //}
//
//    //// Draw solid
//    //float pointSize = 5.0f;
//    //glColor4fv(&solidColor.r);
//    //glPointSize(pointSize);
//    ////glBegin(GL_POINT);
//    ////glVertex3f(0.0f, 0.0f, 0.0f);
//    ////glEnd();
//
//    //// Draw wireframe
//    //float lineLength = 30.0f;
//    //glColor4fv(&wireColor.r);
//    ////glBegin(GL_LINES);
//    ////glVertex3f(lineLength, 0.0f, 0.0f);
//    ////glVertex3f(-lineLength, 0.0f, 0.0f);
//    ////glVertex3f(0.0f, lineLength, 0.0f);
//    ////glVertex3f(0.0f, -lineLength, 0.0f);
//    ////glVertex3f(0.0f, 0.0f, lineLength);
//    ////glVertex3f(0.0f, 0.0f, -lineLength);
//    ////glEnd();
//
//    //glDepthMask(GL_TRUE);
//    //glDisable(GL_BLEND);
//    //glPopAttrib();
//    //view.endGL();
//}

MHWRender::MPxGeometryOverride* MARAP3DHandleLocatorGeometryOverride::creator(const MObject& obj)
{
    return new MARAP3DHandleLocatorGeometryOverride(obj);
}

MHWRender::DrawAPI MARAP3DHandleLocatorGeometryOverride::supportedDrawAPIs() const
{
    return MHWRender::kAllDevices;
}

void MARAP3DHandleLocatorGeometryOverride::updateDG()
{
}

void MARAP3DHandleLocatorGeometryOverride::updateRenderItems(const MDagPath& path, MHWRender::MRenderItemList& list)
{
}

void MARAP3DHandleLocatorGeometryOverride::populateGeometry(const MHWRender::MGeometryRequirements& requirements, const MHWRender::MRenderItemList& renderItems, MHWRender::MGeometry& data)
{
}

void MARAP3DHandleLocatorGeometryOverride::cleanUp()
{
}

bool MARAP3DHandleLocatorGeometryOverride::hasUIDrawables() const
{
    return true;
}

void MARAP3DHandleLocatorGeometryOverride::addUIDrawables(const MDagPath& objPath, MHWRender::MUIDrawManager& drawManager, const MHWRender::MFrameContext& frameContext)
{
    MStatus status = MStatus::kSuccess;
    
    MObject locatorNode = objPath.node();

    float pointSize = 10.0f;
    MColor solidColor(1.0f, 1.0f, 1.0f, 1.0f), wireColor(1.0f, 1.0f, 1.0f, 1.0f);
    unsigned int depthPriority = -1;
    MPoint localPosition = MPoint(0.0f, 0.0f, 0.0f);
    MMatrix worldSpace = MMatrix::identity;

    ////
    //// Get the pointSize attribute value
    ////
    //MPlug pointSizePlug(locatorNode, MARAP3DHandleLocatorNode::aPointSize);
    //status = pointSizePlug.getValue(pointSize);
    //if (!status) {
    //    status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw get pointSize failed !");
    //}
    //
    // Get the worldPosition attribute value
    //
    MPlug localPositionPlug(locatorNode, MARAP3DHandleLocatorNode::localPosition);
    MDataHandle localPositionDataHandle;
    status = localPositionPlug.getValue(localPositionDataHandle);
    if (!status) {
        status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw  get localPositionDataHandle failed !");
    }
    double3& localPositionDouble3 = localPositionDataHandle.asDouble3();
    localPosition = MPoint(localPositionDouble3[0], localPositionDouble3[1], localPositionDouble3[2]);
    if (!status) {
        status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw  get localPosition failed !");
    }
    //
    // Extract the 'worldMatrix' attribute that is inherited from 'dagNode'
    //
    MFnDependencyNode fnLocatorNode(locatorNode);
    MObject worldSpaceAttribute = fnLocatorNode.attribute("worldMatrix");
    MPlug matrixPlug(locatorNode, worldSpaceAttribute);
    //
    // 'worldMatrix' is an array so we must specify which element the plug
    // refers to
    matrixPlug = matrixPlug.elementByLogicalIndex(0);
    //
    // Get the value of the 'worldMatrix' attribute
    //
    MObject matObject;
    status = matrixPlug.getValue(matObject);
    if (!status) {
        status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw get matObject failed !");
    }
    MFnMatrixData matrixData(matObject, &status);
    if (!status) {
        status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw get world matrix data failed !");
    }
    worldSpace = matrixData.matrix(&status);
    if (!status) {
        status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw get world matrix failed !");
    }
    
    
    MHWRender::DisplayStatus displayStatus = MHWRender::MGeometryUtilities::displayStatus(objPath);
    float alpha = 0.6f;
    switch (displayStatus)
    {
    case MHWRender::kLead:
        solidColor = MColor(0.26f, 1.0f, 0.64f, alpha);
        wireColor = MColor(0.26f, 1.0f, 0.64f, 1.0f);
        depthPriority = MHWRender::MRenderItem::sSelectionDepthPriority;
        break;
    case MHWRender::kActive:
        solidColor = MColor(1.0f, 1.0f, 1.0f, alpha);
        wireColor = MColor(1.0f, 1.0f, 1.0f, 1.0f);
        depthPriority = MHWRender::MRenderItem::sActivePointDepthPriority;
        break;
    default:
        solidColor = MColor(1.0f, 1.0f, 0.0f, alpha);
        wireColor = MColor(1.0f, 1.0f, 0.0f, 1.0f);
        depthPriority = MHWRender::MRenderItem::sDormantPointDepthPriority;
        break;
    }

    unsigned int previousDepthPriority = drawManager.depthPriority();

    drawManager.setDepthPriority(depthPriority);

    drawManager.beginDrawable(MHWRender::MUIDrawManager::kSelectable, 20);
    drawManager.setColor(solidColor);
    drawManager.setPointSize(pointSize);

    drawManager.point(localPosition);
    //drawManager.point(localPosition * worldSpace);
    drawManager.endDrawable();

    drawManager.setDepthPriority(previousDepthPriority);
}

MARAP3DHandleLocatorGeometryOverride::MARAP3DHandleLocatorGeometryOverride(const MObject& obj)
    : MHWRender::MPxGeometryOverride(obj)
{
}

//MPxDrawOverride* MARAP3DHandleLocatorDrawOverride::creator(const MObject& obj)
//{
//    return new MARAP3DHandleLocatorDrawOverride(obj);
//}
//
//MHWRender::DrawAPI MARAP3DHandleLocatorDrawOverride::supportedDrawAPIs() const
//{ 
//    return MHWRender::kAllDevices; 
//}
//
//bool MARAP3DHandleLocatorDrawOverride::isBounded(const MDagPath& objPath, const MDagPath& cameraPath) const
//{
//    return false;
//}
//
//MBoundingBox MARAP3DHandleLocatorDrawOverride::boundingBox(const MDagPath& objPath, const MDagPath& cameraPath) const
//{
//    return MPxDrawOverride::boundingBox(objPath, cameraPath);
//}
//
//bool MARAP3DHandleLocatorDrawOverride::isTransparent() const
//{
//    return true;
//}
//
//bool MARAP3DHandleLocatorDrawOverride::wantUserSelection() const
//{
//    return false; // true only if override userSelection().
//}
//
//MUserData* MARAP3DHandleLocatorDrawOverride::prepareForDraw(const MDagPath& objPath, const MDagPath& cameraPath, const MFrameContext& frameContext, MUserData* oldData)
//{
//    // Called by Maya whenever the object is dirty and needs to update for draw.
//    // Any data needed from the Maya dependency graph must be retrieved and cached
//    // in this stage. It is invalid to pull data from the Maya dependency graph in
//    // the draw callback method and Maya may become unstable if that is attempted.
//
//    MStatus status = MStatus::kSuccess;
//
//    // Retrieve data cache (create if does not exist)
//    MARAP3DHandleLocatorDrawOverrideData* overrideData = dynamic_cast<MARAP3DHandleLocatorDrawOverrideData*>(oldData);
//    if(!overrideData)
//    {
//        overrideData = new MARAP3DHandleLocatorDrawOverrideData();
//    }
//
//    MObject locatorNode = objPath.node();
//    ////
//    //// Get the pointSize attribute value
//    ////
//    //MPlug pointSizePlug(locatorNode, MARAP3DHandleLocatorNode::aPointSize);
//    //status = pointSizePlug.getValue(overrideData->pointSize);
//    //if (!status) {
//    //    status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw get pointSize failed !");
//    //}
//    //
//    // Get the worldPosition attribute value
//    //
//    MPlug localPositionPlug(locatorNode, MARAP3DHandleLocatorNode::localPosition);
//    MDataHandle localPositionDataHandle;
//    status = localPositionPlug.getValue(localPositionDataHandle);
//    if (!status) {
//        status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw  get localPositionDataHandle failed !");
//    }
//    double3& localPositionDouble3 = localPositionDataHandle.asDouble3();
//    overrideData->localPosition = MPoint(localPositionDouble3[0], localPositionDouble3[1], localPositionDouble3[2]);
//    if (!status) {
//        status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw  get localPosition failed !");
//    }
//    //
//    // Extract the 'worldMatrix' attribute that is inherited from 'dagNode'
//    //
//    MFnDependencyNode fnLocatorNode(locatorNode);
//    MObject worldSpaceAttribute = fnLocatorNode.attribute("worldMatrix");
//    MPlug matrixPlug(locatorNode, worldSpaceAttribute);
//    //
//    // 'worldMatrix' is an array so we must specify which element the plug
//    // refers to
//    matrixPlug = matrixPlug.elementByLogicalIndex(0);
//    //
//    // Get the value of the 'worldMatrix' attribute
//    //
//    MObject matObject;
//    status = matrixPlug.getValue(matObject);
//    if (!status) {
//        status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw get matObject failed !");
//    }
//    MFnMatrixData matrixData(matObject, &status);
//    if (!status) {
//        status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw get world matrix data failed !");
//    }
//    overrideData->worldSpace = matrixData.matrix(&status);
//    if (!status) {
//        status.perror("MARAP3DHandleLocatorDrawOverride::prepareForDraw get world matrix failed !");
//    }
//
//
//    MHWRender::DisplayStatus displayStatus = MHWRender::MGeometryUtilities::displayStatus(objPath);
//    float alpha = 0.6f;
//    switch (displayStatus)
//    {
//    case MHWRender::kLead:
//        overrideData->solidColor = MColor(0.26f, 1.0f, 0.64f, alpha);
//        overrideData->wireColor = MColor(0.26f, 1.0f, 0.64f, 1.0f);
//        overrideData->depthPriority = MHWRender::MRenderItem::sSelectionDepthPriority;
//        break;
//    case MHWRender::kActive:
//        overrideData->solidColor = MColor(1.0f, 1.0f, 1.0f, alpha);
//        overrideData->wireColor = MColor(1.0f, 1.0f, 1.0f, 1.0f);
//        overrideData->depthPriority = MHWRender::MRenderItem::sActivePointDepthPriority;
//        break;
//    default:
//        overrideData->solidColor = MColor(1.0f, 1.0f, 0.0f, alpha);
//        overrideData->wireColor = MColor(1.0f, 1.0f, 0.0f, 1.0f);
//        overrideData->depthPriority = MHWRender::MRenderItem::sDormantPointDepthPriority;
//        break;
//    }
//
//    return overrideData;
//}
//
//bool MARAP3DHandleLocatorDrawOverride::hasUIDrawables() const
//{
//    return true;
//}
//
//void MARAP3DHandleLocatorDrawOverride::addUIDrawables(const MDagPath& objPath, MHWRender::MUIDrawManager& drawManager, const MHWRender::MFrameContext& frameContext, const MUserData* data)
//{
//    MStatus status = MStatus::kSuccess;
//
//    // Retrieve data cache (create if does not exist)
//    const MARAP3DHandleLocatorDrawOverrideData* overrideData = static_cast<const MARAP3DHandleLocatorDrawOverrideData*>(data);
//    if (!overrideData)
//    {
//        return;
//    }
//
//    unsigned int previousDepthPriority = drawManager.depthPriority();
//
//    drawManager.setDepthPriority(overrideData->depthPriority);
//
//    drawManager.beginDrawable(MHWRender::MUIDrawManager::kSelectable, 1);
//    drawManager.setColor(overrideData->solidColor);
//    drawManager.setPointSize(overrideData->pointSize);
//
//    drawManager.point(overrideData->localPosition);
//    //drawManager.point(overrideData->localPosition * overrideData->worldSpace);
//    drawManager.endDrawable();
//
//    drawManager.setDepthPriority(previousDepthPriority);
//}
//
//MARAP3DHandleLocatorDrawOverride::MARAP3DHandleLocatorDrawOverride(const MObject& obj)
//    : MHWRender::MPxDrawOverride(obj, nullptr, false/*isAlwaysDirty*/)
//{
//}
