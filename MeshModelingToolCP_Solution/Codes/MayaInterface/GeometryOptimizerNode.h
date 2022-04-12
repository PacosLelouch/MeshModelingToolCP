#pragma once

#include "MayaNodeCommon.h"
#include <maya/MPxDeformerNode.h>
#include <maya/MMatrix.h>
#include <maya/MFnMesh.h>
#include "LGSolver/Solvers/GeometrySolver.h"
#include "MayaToEigenConverter.h"

// The abstract class of the geometry optimizer node. Don't create creator!
class MGeometryOptimizerNode : public MPxDeformerNode
{
public:
//    static MStatus initialize();

    static MStatus jumpToElement(MArrayDataHandle& hArray, unsigned int index);

    static bool isMeshNotAssigned(const MObject& meshToAssign, const MObject& meshInput);
    static bool isMeshDirty(const MObject& meshToCompare, const MObject& meshInput);

public:
    virtual void postConstructor() override;

    MObject getMeshObjectFromInputWithoutEval(MDataBlock& block, unsigned int index, MStatus* statusPtr = nullptr);
    MObject getMeshObjectFromInput(MDataBlock& block, unsigned int index, MStatus* statusPtr = nullptr);
    MObject getMeshObjectFromOutput(MDataBlock& block, unsigned int index, MStatus* statusPtr = nullptr);

//public:
//    static MObject aTime;
//    static MObject aNumIter;
//
//    static bool initialized;

public:
#if _COMPUTE_USING_CUDA
    //TODO: AAShapeUp::AndersonAccelerationOptimizerGPU<3>, Jacobi_LinearSolverGPU<3>
    using MyGeometrySolver3D = AAShapeUp::GeometrySolver<3, AAShapeUp::OpenMPTimer, AAShapeUp::AndersonAccelerationOptimizer<3>, AAShapeUp::Simplicial_LDLT_LinearSolver<3>>;
#else // !_COMPUTE_USING_CUDA
    using MyGeometrySolver3D = AAShapeUp::GeometrySolver<3, AAShapeUp::OpenMPTimer, AAShapeUp::AndersonAccelerationOptimizer<3>, AAShapeUp::Simplicial_LDLT_LinearSolver<3>>;
#endif // _COMPUTE_USING_CUDA

    std::shared_ptr<MyGeometrySolver3D> m_geometrySolverShPtr;

    std::shared_ptr<AAShapeUp::MayaToEigenConverter> m_meshConverterShPtr;
};
