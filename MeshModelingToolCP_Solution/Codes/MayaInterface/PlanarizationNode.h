#pragma once

#include "GeometryOptimizerNode.h"
#include "Operations/PlanarizationOperation.h"

struct PlanarizationCache
{
    int numIter = -1;
    double maxDisplacementVisualization = -1.0;
    double planarityWeight = -1.0;
    double closenessWeight = -1.0;
    double fairnessWeight = -1.0;
    double relativeFairnessWeight = -1.0;
    MObject inputMeshObj = MObject::kNullObj;
    MObject referenceMeshObj = MObject::kNullObj;
};

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

    PlanarizationCache m_cache;

public:
    static const MTypeId id;
    static const MString nodeName;

    static MObject aNumIter;
    static MObject aMaxDisplacementVisualization;
    static MObject aPlanarityWeight;
    static MObject aClosenessWeight;
    static MObject aFairnessWeight;
    static MObject aRelativeFairnessWeight;

    static MObject aReferenceMesh;
};

