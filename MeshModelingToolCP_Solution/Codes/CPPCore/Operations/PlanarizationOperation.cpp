#include "pch.h"
#include "PlanarizationOperation.h"

BEGIN_NAMESPACE(AAShapeUp)

bool PlanarizationOperation::initializeConstraintsAndRegularizations()
{
    //TODO: Initialize constraints and regularizations.

    return true;
}

MeshDirtyFlag PlanarizationOperation::getPlanarityErrors(std::vector<scalar>& outErrors, scalar maxError) const
{
    //TODO: Generate planarity error as color.
    return MeshDirtyFlag::ColorDirty;
}

MeshDirtyFlag PlanarizationOperation::getMeshDirtyFlag() const
{ 
    return MeshDirtyFlag::PositionDirty; 
}

END_NAMESPACE()
