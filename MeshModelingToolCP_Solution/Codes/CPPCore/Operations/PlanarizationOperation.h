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

    virtual MeshDirtyFlag getPlanarityErrors(std::vector<scalar>& outErrors, scalar maxError = 1) const override;

    virtual MeshDirtyFlag getMeshDirtyFlag() const override;

protected:
};

END_NAMESPACE()
