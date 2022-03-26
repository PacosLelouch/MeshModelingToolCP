#pragma once

#include "OperationBase.h"

BEGIN_NAMESPACE(AAShapeUp)

class PlanarizationOperation : public OperationBase
{
public:
    using Super = OperationBase;

    PlanarizationOperation(const std::shared_ptr<SolverAbstract<3, LinearRegularizer<3>, ConstraintSet<3>>>& solverShPtr)
        : Super(solverShPtr) {}

    virtual ~PlanarizationOperation() {}

    virtual bool initializeConstraintsAndRegularizations() override;

    virtual MeshDirtyFlag getOutputErrors(std::vector<scalar>& outErrors, scalar maxError = 1) const override;

    virtual MeshDirtyFlag getMeshDirtyFlag() const override;


    EigenMesh<3> refMesh;
    scalar closeness_weight, relative_laplacian_weight, laplacian_weight, planarity_weight;

protected:
};

END_NAMESPACE()
