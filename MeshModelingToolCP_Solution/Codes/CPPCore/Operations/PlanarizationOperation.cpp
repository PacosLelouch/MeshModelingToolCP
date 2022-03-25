#include "pch.h"
#include "PlanarizationOperation.h"

BEGIN_NAMESPACE(AAShapeUp)

bool PlanarizationOperation::initialize(const std::vector<i32>& vertexIndices, const std::vector<i32>& numFaceVertices, const Matrix3X& positions, const std::vector<i32>& handleIndices)
{
    m_vertexIndices = vertexIndices;
    m_numFaceVertices = numFaceVertices;
    m_positions = positions;
    m_handleIndices = handleIndices;
    //TODO: init constraints and regularizations.

    return m_solverShPtr && m_solverShPtr->initialize(static_cast<i32>(positions.cols()), m_handleIndices);
}

bool PlanarizationOperation::solve(Matrix3X& newPositions, i32 nIter)
{
    if (!m_solverShPtr->solve(nIter, &m_positions))
    {
        return false;
    }
    m_solverShPtr->getOutput(newPositions);
    return true;
}

END_NAMESPACE()
