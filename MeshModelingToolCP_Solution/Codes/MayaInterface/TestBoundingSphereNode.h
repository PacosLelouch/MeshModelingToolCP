#pragma once

#include "GeometryOptimizerNode.h"
#include "Operations/TestBoundingSphereOperation.h"

struct TestBoundingSphereCache
{
    int numIter = -1;
    double sphereProjectionWeight = -1.0;
    double fairnessWeight = -1.0;
    MObject inputMeshObj = MObject::kNullObj;
};

class MTestBoundingSphereNode : public MGeometryOptimizerNode
{
public:
    using Super = MGeometryOptimizerNode;
public:
    static void* creator();
    static MStatus initialize();
public:
    virtual MStatus compute(const MPlug& plug, MDataBlock& block) override;

    // when the accessory is deleted, this node will clean itself up
    //
    virtual MObject& accessoryAttribute() const override;
    // create accessory nodes when the node is created
    //
    virtual MStatus accessoryNodeSetup(MDagModifier& cmd) override;

    virtual MStatus deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex) override;

public:

    std::shared_ptr<AAShapeUp::TestBoundingSphereOperation> m_operationShPtr;
    TestBoundingSphereCache m_cache;

public:
    static const MTypeId id;
    static const MString nodeName;

    static MObject aNumIter;
    static MObject aSphereProjectionWeight;
    static MObject aFairnessWeight;
};

