#include "pch.h"
#include "MinimalSurfaceOperation.h"
#include "LGSolver/Constraints/PointConstraint.h"
#include "LGSolver/Regularizations/LaplacianRegTerm.h"
#include <unordered_set>

BEGIN_NAMESPACE(AAShapeUp)

bool MinimalSurfaceOperation::initializeConstraintsAndRegularizations()
{
    // Initialize constraints and regularizations.

    std::unordered_set<i32> boundaryVertexSet;
    this->m_mesh.m_section.getBoundaryVertexSet(boundaryVertexSet);

    // Constraints.
    std::unordered_set<i32> handleSet(this->m_handleIndices.begin(), this->m_handleIndices.end());

    for (i64 vidx = 0; vidx < this->m_initialPositions.cols(); ++vidx)
    {
        if (handleSet.find(i32(vidx)) == handleSet.end() && boundaryVertexSet.find(i32(vidx)) == boundaryVertexSet.end())
        {
            continue;
        }
        this->m_solverShPtr->addConstraint(std::make_shared<PointConstraint<3>>(vidx, this->m_fixBoundaryWeight, this->m_initialPositions.col(vidx)));
    }

    // Regularizations.
    std::unordered_map<i32, std::unordered_set<i32>> vertexAdjacentVerticesMap;
    this->m_mesh.m_section.getVertexAdjacentVerticesMap(vertexAdjacentVerticesMap);

    for (auto& vertexAdjacentVerticesPair : vertexAdjacentVerticesMap)
    {
        i32 fromIdx = vertexAdjacentVerticesPair.first;
        if (boundaryVertexSet.find(fromIdx) != boundaryVertexSet.end())
        {
            continue;
        }
        std::unordered_set<i32>& toIdxs = vertexAdjacentVerticesPair.second;
        std::vector<i32> indices(1, fromIdx);
        indices.insert(indices.end(), toIdxs.begin(), toIdxs.end());
        //this->m_solverShPtr->addRegularizationTerm(std::make_shared<UniformLaplacianRelativeRegTerm<3>>(indices, this->m_LaplacianWeight, this->m_initialPositions));
        this->m_solverShPtr->addRegularizationTerm(std::make_shared<UniformLaplacianRegTerm<3>>(indices, this->m_LaplacianWeight));
    }

    return true;
}

std::tuple<MeshDirtyFlag, MeshIndexType> MinimalSurfaceOperation::getOutputErrors(std::vector<scalar>& outErrors) const
{
    Matrix3X finalPositions;
    this->m_solverShPtr->getOutput(finalPositions);

    std::unordered_set<i32> boundaryVertexSet;
    this->m_mesh.m_section.getBoundaryVertexSet(boundaryVertexSet);

    std::unordered_map<i32, std::unordered_set<i32>> vertexAdjacentVerticesMap;
    this->m_mesh.m_section.getVertexAdjacentVerticesMap(vertexAdjacentVerticesMap);

    VectorX sumNum;
    sumNum.setZero(finalPositions.cols());
    Matrix3X uniformNormals;
    uniformNormals.setZero(finalPositions.rows(), finalPositions.cols());

    for (auto& vertexAdjacentVerticesPair : vertexAdjacentVerticesMap)
    {
        i32 i = vertexAdjacentVerticesPair.first;
        if (boundaryVertexSet.find(i) != boundaryVertexSet.end())
        {
            continue;
        }
        for (i32 j : vertexAdjacentVerticesPair.second)
        {
            uniformNormals.col(i) += finalPositions.col(j) - finalPositions.col(i);
            sumNum(i) += scalar(1.0);
        }
    }

    outErrors.resize(finalPositions.cols());

    for (i64 i = 0; i < finalPositions.cols(); ++i)
    {
        scalar sum = sumNum(i);
        outErrors[i] = (boundaryVertexSet.find(i) != boundaryVertexSet.end() || sum == scalar(0.0)) ? scalar(0.0) : uniformNormals.col(i).norm() / sumNum(i);
    }

    return { MeshDirtyFlag::ColorDirty, MeshIndexType::PerVertex };
}

MeshDirtyFlag MinimalSurfaceOperation::getMeshDirtyFlag() const
{ 
    return MeshDirtyFlag::PositionDirty; 
}

END_NAMESPACE()
