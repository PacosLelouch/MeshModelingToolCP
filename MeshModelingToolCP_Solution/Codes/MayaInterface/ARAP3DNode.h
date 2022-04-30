#pragma once

#include <maya/MMatrixArray.h>
#include <maya/MIntArray.h>
#include "GeometryOptimizerNode.h"
#include "Operations/ARAP3DOperation.h"

struct ARAP3DCache
{
    int numIter = -1;
    double maxDisplacementVisualization = -1.0;
    double deformationWeight = -1.0;
    std::vector<MPoint> handlePositions;
    std::vector<int> handleIndices;
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
    virtual void postConstructor() override;
    virtual MStatus deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex) override;

public:

    std::shared_ptr<AAShapeUp::ARAP3DOperation> m_operationShPtr; 

    ARAP3DCache m_cache;

public:
    static const MTypeId id;
    static const MString nodeName;

    static MObject aNumIter;
    static MObject aMaxDisplacementVisualization;
    static MObject aDeformationWeight;
    static MObject aHandlePositions;
    static MObject aHandleIndices;
};

