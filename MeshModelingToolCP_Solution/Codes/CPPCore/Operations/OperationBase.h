#pragma once

#include "LGSolver/Solvers/GeometrySolver.h"
#include "EigenMesh.h"

BEGIN_NAMESPACE(AAShapeUp)

class OperationBase
{
public:
    OperationBase(const std::shared_ptr<SolverAbstract<3, LinearRegularizer<3>, ConstraintSet<3>>>& solverShPtr)
        : m_solverShPtr(solverShPtr) {}

    virtual ~OperationBase() {}

    bool initialize(const std::vector<i32>& vertexIndices, const std::vector<i32>& numFaceVertices, const Matrix3X& positions, const std::vector<i32>& handleIndices);

    virtual bool initializeConstraintsAndRegularizations() = 0;

    virtual bool solve(Matrix3X& newPositions, i32 nIter = 5);

    MeshDirtyFlag visualizeOutputErrors(Matrix3X& outColors, scalar maxError = 1) const;

    virtual MeshDirtyFlag getOutputErrors(std::vector<scalar>& outErrors, scalar maxError = 1) const = 0;

    virtual MeshDirtyFlag getMeshDirtyFlag() const = 0;

    // HDR, useless?
    static void ReinhardOperatorBatch(Matrix3X& inOutColors);

    static void HSV2RGBBatch(Matrix3X& inOutColors);

    static Vector3 HSV2RGB(Vector3 inHSV);

protected:
    std::shared_ptr<SolverAbstract<3, LinearRegularizer<3>, ConstraintSet<3>>> m_solverShPtr;

    std::vector<i32> m_handleIndices;

    std::vector<i32> m_vertexIndices;
    std::vector<i32> m_numFaceVertices;

    Matrix3X m_initialPositions;
};

END_NAMESPACE()
