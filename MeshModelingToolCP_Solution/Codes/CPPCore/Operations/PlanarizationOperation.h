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
    scalar closeness_weight = scalar(1), relative_laplacian_weight = scalar(0.1), laplacian_weight = scalar(0.1), planarity_weight = scalar(1);

protected:
};

END_NAMESPACE()
