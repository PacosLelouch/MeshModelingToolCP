#include "pch.h"
#include "LGSolver/Constraints/TestBoundingSphereConstraint.h"
#include "TestBoundingSphereOperation.h"
#include "LGSolver/Regularizations/LaplacianRegTerm.h"
#include <unordered_set>

BEGIN_NAMESPACE(AAShapeUp)

bool TestBoundingSphereOperation::initializeConstraintsAndRegularizations()
{
    // Initialize constraints and regularizations.

    // Constraints.
    std::unordered_set<i32> handleSet(this->m_handleIndices.begin(), this->m_handleIndices.end());

    m_radius = scalar(0);
    scalar invSize = scalar(1) / this->m_initialPositions.cols();
    m_center = this->m_initialPositions.col(0) * invSize;
    m_maxCorner = m_center;

    for (i64 vidx = 1; vidx < this->m_initialPositions.cols(); ++vidx)
    {
        Vector3 curPos = this->m_initialPositions.col(vidx);
        m_center += curPos * invSize;
        for (i64 i = 0; i < m_maxCorner.size(); ++i)
        {
            m_maxCorner(i) = glm::max(m_maxCorner(i), curPos(i));
        }
    }
    m_radius = (m_maxCorner - m_center).norm();

    for (i64 vidx = 0; vidx < this->m_initialPositions.cols(); ++vidx)
    {
        if (handleSet.find(i32(vidx)) != handleSet.end())
        {
            continue;
        }
        this->m_solverShPtr->addConstraint(std::make_shared<TestBoundingSphereConstraint<3>>(vidx, this->m_sphereProjectionWeight, m_center, m_radius));
    }

    // Regularizations.
    std::unordered_map<i32, std::unordered_set<i32>> vertexAdjacentVerticesMap;
    this->m_mesh.m_section.getVertexAdjacentVerticesMap(vertexAdjacentVerticesMap);

    std::unordered_set<i32> boundaryVertexSet;
    this->m_mesh.m_section.getBoundaryVertexSet(boundaryVertexSet);

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

std::tuple<MeshDirtyFlag, MeshIndexType> TestBoundingSphereOperation::getOutputErrors(std::vector<scalar>& outErrors) const
{
    Matrix3X finalPositions;
    this->m_solverShPtr->getOutput(finalPositions);

    outErrors.resize(finalPositions.cols());

    for (i64 i = 0; i < finalPositions.cols(); ++i)
    {
        Vector3 cur = finalPositions.col(i);
        Vector3 target = m_center + (finalPositions.col(i) - m_center).normalized() * m_radius;
        outErrors[i] = (cur - target).norm();
    }

    return { MeshDirtyFlag::ColorDirty, MeshIndexType::PerVertex };
}

MeshDirtyFlag TestBoundingSphereOperation::getMeshDirtyFlag() const
{ 
    return MeshDirtyFlag::PositionDirty; 
}

END_NAMESPACE()
