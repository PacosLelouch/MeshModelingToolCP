#pragma once

#include "GeometryOptimizerNode.h"
//#include "Operations/PlanarizationOperation.h"

struct ARAP3DCache
{
    int numIter = -1;
    MObject inputMeshObj = MObject::kNullObj;
};

class MARAP3DNode : public MGeometryOptimizerNode
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
    //std::shared_ptr<AAShapeUp::PlanarizationOperation> m_operationShPtr; // TODO

    ARAP3DCache m_cache;

public:
    static const MTypeId id;
    static const MString nodeName;

    static MObject aNumIter;
};

