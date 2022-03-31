#pragma once

#include "GeometryOptimizerNode.h"
#include "Operations/PlanarizationOperation.h"

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

    std::shared_ptr<AAShapeUp::MayaToEigenConverter> m_meshConverterReferenceShPtr;
    std::shared_ptr<AAShapeUp::PlanarizationOperation> m_operationShPtr;

public:
    static const MTypeId id;
    static const MString nodeName;

    static MObject aNumIter;
    static MObject aPlanarityWeight;
    static MObject aClosenessWeight;
    static MObject aFairnessWeight;
    static MObject aRelativeFairnessWeight;

    static MObject aReferenceMesh;
};

