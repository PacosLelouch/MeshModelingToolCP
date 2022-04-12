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

    bool initialize(const EigenMesh<3>& mesh, const std::vector<i32>& handleIndices);

    virtual bool initializeConstraintsAndRegularizations() = 0;

    virtual bool solve(Matrix3X& newPositions, i32 nIter = 5);

    MeshDirtyFlag visualizeOutputErrors(Matrix3X& outColors, scalar maxError = 1, bool keepHeatValue = false) const;

    virtual std::tuple<MeshDirtyFlag, MeshIndexType> getOutputErrors(std::vector<scalar>& outErrors) const = 0;

    MeshDirtyFlag visualizeDisplacements(Matrix3X& outColors, scalar maxDisplacement = 1, bool keepHeatValue = false) const;

    void getDisplacements(Matrix3X& displacements) const;

    virtual MeshDirtyFlag getMeshDirtyFlag() const = 0;

    // HDR, useless?
    static void ReinhardOperatorBatch(Matrix3X& inOutColors);

    static void HSV2RGBBatch(Matrix3X& inOutColors);

    static Vector3 HSV2RGB(Vector3 inHSV);

protected:
    void trianglesToVertices(std::vector<scalar>& outValues, const std::vector<scalar>& inValues) const;
    void polygonsToVertices(std::vector<scalar>& outValues, const std::vector<scalar>& inValues) const;

protected:
    std::shared_ptr<SolverAbstract<3, LinearRegularizer<3>, ConstraintSet<3>>> m_solverShPtr;

    std::vector<i32> m_handleIndices;

    EigenMesh<3> m_mesh;
    //std::vector<i32> m_vertexIndices;
    //std::vector<i32> m_numFaceVertices;

    Matrix3X m_initialPositions;
};

END_NAMESPACE()
