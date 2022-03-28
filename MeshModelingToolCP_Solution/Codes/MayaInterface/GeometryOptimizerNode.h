#pragma once

#include <maya/MPxDeformerNode.h>
#include <maya/MMatrix.h>
#include <maya/MFnMesh.h>
#include "LGSolver/Solvers/GeometrySolver.h"

// The abstract class of the geometry optimizer node. Don't create creator!
class MGeometryOptimizerNode : public MPxDeformerNode
{
public:
    static MStatus initialize();

    static MStatus jumpToElement(MArrayDataHandle& hArray, unsigned int index);

public:
    MFnMesh getFnMeshFromInput(MDataBlock& block);

public:
    static MObject aTime;
    static bool initialized;

public:
    using MyGeometrySolver3D = AAShapeUp::GeometrySolver<3, AAShapeUp::OpenMPTimer, AAShapeUp::AndersonAccelerationOptimizer<3>, AAShapeUp::Simplicial_LDLT_LinearSolver<3>>;

    std::shared_ptr<MyGeometrySolver3D> mGeometrySolverShPtr;
};
