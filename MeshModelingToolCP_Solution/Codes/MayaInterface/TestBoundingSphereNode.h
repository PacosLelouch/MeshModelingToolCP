#pragma once

#include "GeometryOptimizerNode.h"
#include "Operations/TestBoundingSphereOperation.h"

class MTestBoundingSphereNode : public MGeometryOptimizerNode
{
public:
    using Super = MGeometryOptimizerNode;
public:
    static void* creator();
    static MStatus initialize();
public:
    virtual MStatus deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex) override;

public:

    std::shared_ptr<AAShapeUp::TestBoundingSphereOperation> m_operationShPtr;

public:
    static const MTypeId id;
    static const MString nodeName;

    static MObject aNumIter;
    static MObject aSphereProjectionWeight;
    static MObject aFairnessWeight;
};

