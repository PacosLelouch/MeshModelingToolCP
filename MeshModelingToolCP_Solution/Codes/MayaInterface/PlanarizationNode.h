#pragma once

#include "GeometryOptimizerNode.h"

class MPlanarizationNode : public MGeometryOptimizerNode
{
public:
    using Super = MGeometryOptimizerNode;
public:
    static void* creator();
    static MStatus initialize();
public:
    virtual MStatus deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex) override;

public:
    static const MTypeId id;
    static const MString nodeName;
};

