#pragma once

#include "LGSolver/Solvers/GeometrySolver.h"

BEGIN_NAMESPACE(AAShapeUp)

class PlanarizationOperation
{
public:
    PlanarizationOperation(const std::shared_ptr<SolverAbstract<3, LinearRegularizer<3>, ConstraintSet<3>>>& solverShPtr)
        : m_solverShPtr(solverShPtr) {}

    bool initialize(const std::vector<i32>& vertexIndices, const std::vector<ui8>& numFaceVertices, const Matrix3X& positions, const std::vector<i32>& handleIndices);

    bool solve(Matrix3X& newPositions, i32 nIter = 5);

protected:
    std::shared_ptr<SolverAbstract<3, LinearRegularizer<3>, ConstraintSet<3>>> m_solverShPtr;

    std::vector<i32> m_handleIndices;

    std::vector<i32> m_vertexIndices;
    std::vector<ui8> m_numFaceVertices;

    Matrix3X m_positions;
};

END_NAMESPACE()
