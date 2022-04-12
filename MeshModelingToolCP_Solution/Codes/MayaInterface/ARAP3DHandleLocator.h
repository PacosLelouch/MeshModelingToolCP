#pragma once

#include "MayaNodeCommon.h"
#include <maya/MPxLocatorNode.h>
// Viewport 2.0 includes
#include <maya/MDrawRegistry.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MPxGeometryOverride.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MHWGeometryUtilities.h>
#include <maya/MPointArray.h>
#include <maya/MGlobal.h>
#include <maya/MEventMessage.h>
#include <maya/MFnDependencyNode.h>

class MARAP3DHandleLocatorNode : public MPxLocatorNode
{
public:
    static void* creator();
    static MStatus initialize();

public:
    virtual void postConstructor() override;

    virtual bool useClosestPointForSelection() const override;

    //virtual bool isBounded() const override;

    //virtual bool isTransparent() const override;

    //virtual void draw(M3dView & view, const MDagPath & path, M3dView::DisplayStyle style, M3dView::DisplayStatus status) override;

public:
    static const MTypeId id;
    static const MString nodeName;

    static const MString drawDbClassification;
    static const MString drawRegistrantId;

    //static MObject aPointSize;
    static MObject aVertexIndex;

public:

};

class MARAP3DHandleLocatorGeometryOverride : public MHWRender::MPxGeometryOverride
{
public:
    static MPxGeometryOverride* creator(const MObject& obj);

    virtual MHWRender::DrawAPI supportedDrawAPIs() const override;

    virtual void updateDG() override;
    virtual void updateRenderItems(const MDagPath& path, MHWRender::MRenderItemList& list) override;
    virtual void populateGeometry(const MHWRender::MGeometryRequirements& requirements,
        const MHWRender::MRenderItemList& renderItems,
        MHWRender::MGeometry& data) override;
    virtual void cleanUp();

    // Return true so addUIDrawables() will be called by the Viewport 2.0. 
    virtual bool hasUIDrawables() const override;
    virtual void addUIDrawables(const MDagPath& objPath, MHWRender::MUIDrawManager& drawManager, const MHWRender::MFrameContext& frameContext) override;
protected:
    MARAP3DHandleLocatorGeometryOverride(const MObject& obj);
};

//
// Custom MUserData to cache the data used.
//
class MARAP3DHandleLocatorDrawOverrideData : public MUserData
{
public:
    MARAP3DHandleLocatorDrawOverrideData() : MUserData(false) {} // don't delete after draw
    virtual ~MARAP3DHandleLocatorDrawOverrideData() override {}

    bool enableDrawing = true;
    MPoint localPosition = MPoint(0.0, 0.0, 0.0, 1.0);
    MMatrix worldSpace = MMatrix::identity;
    float pointSize = 10.0f;
    unsigned int depthPriority = 0;
    MColor solidColor, wireColor;
};
////
//// Implementation of custom MPxDrawOverride to draw in the Viewport 2.0.
////
//class MARAP3DHandleLocatorDrawOverride : public MHWRender::MPxDrawOverride
//{
//public:
//    static MPxDrawOverride* creator(const MObject& obj);
//
//    virtual MHWRender::DrawAPI supportedDrawAPIs() const override;
//    virtual bool isBounded(const MDagPath& objPath, const MDagPath& cameraPath) const override;
//    virtual MBoundingBox boundingBox(const MDagPath& objPath, const MDagPath& cameraPath) const override;
//    virtual bool isTransparent() const override;
//    virtual bool wantUserSelection() const override;
//    virtual MUserData* prepareForDraw(const MDagPath& objPath, const MDagPath& cameraPath, const MFrameContext& frameContext, MUserData* oldData) override;
//    // Return true so addUIDrawables() will be called by the Viewport 2.0. 
//    virtual bool hasUIDrawables() const override;
//    virtual void addUIDrawables(const MDagPath& objPath, MHWRender::MUIDrawManager& drawManager, const MHWRender::MFrameContext& frameContext, const MUserData* data) override;
//private:
//    MARAP3DHandleLocatorDrawOverride(const MObject& obj);
//};
