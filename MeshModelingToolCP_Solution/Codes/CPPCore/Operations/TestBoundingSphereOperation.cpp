#include "pch.h"
#include "LGSolver/Constraints/TestBoundingSphereConstraint.h"
#include "TestBoundingSphereOperation.h"
#include <unordered_set>

BEGIN_NAMESPACE(AAShapeUp)

bool TestBoundingSphereOperation::initializeConstraintsAndRegularizations()
{
    //TODO: Initialize constraints and regularizations.
    std::unordered_set<i32> handleSet(this->m_handleIndices.begin(), this->m_handleIndices.end());
    scalar radius = scalar(0);
    scalar invSize = scalar(1) / this->m_initialPositions.cols();
    Vector3 center = this->m_initialPositions.col(0) * invSize;
    Vector3 maxCorner = center;

    for (i64 vidx = 1; vidx < this->m_initialPositions.cols(); ++vidx)
    {
        Vector3 curPos = this->m_initialPositions.col(vidx);
        center += curPos * invSize;
        for (i64 i = 0; i < maxCorner.size(); ++i)
        {
            maxCorner(i) = glm::max(maxCorner(i), curPos(i));
        }
    }
    //TODO:radius = maxCorner - 

    for (i64 vidx = 0; vidx < this->m_initialPositions.cols(); ++vidx)
    {
        if (handleSet.find(i32(vidx)) != handleSet.end())
        {
            continue;
        }
        this->m_solverShPtr->addConstraint(std::make_shared<TestBoundingSphereConstraint<3>>(vidx, this->m_weight, center, radius));
    }
    return true;
}

MeshDirtyFlag TestBoundingSphereOperation::getPlanarityErrors(std::vector<scalar>& outErrors, scalar maxError) const
{
    //TODO: Generate planarity error as color.
    return MeshDirtyFlag::ColorDirty;
}

MeshDirtyFlag TestBoundingSphereOperation::getMeshDirtyFlag() const
{ 
    return MeshDirtyFlag::PositionDirty; 
}

END_NAMESPACE()
