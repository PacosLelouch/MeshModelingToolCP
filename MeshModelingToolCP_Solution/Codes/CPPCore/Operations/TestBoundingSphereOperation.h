#pragma once

#include "OperationBase.h"

BEGIN_NAMESPACE(AAShapeUp)

class TestBoundingSphereOperation : public OperationBase
{
public:
    using Super = OperationBase;

    TestBoundingSphereOperation(const std::shared_ptr<SolverAbstract<3, LinearRegularizer<3>, ConstraintSet<3>>>& solverShPtr, scalar weight = scalar(1))
        : Super(solverShPtr)
        , m_weight(weight) {}

    virtual ~TestBoundingSphereOperation() {}

    virtual bool initializeConstraintsAndRegularizations() override;

    virtual MeshDirtyFlag getOutputErrors(std::vector<scalar>& outErrors, scalar maxError = 1) const override;

    virtual MeshDirtyFlag getMeshDirtyFlag() const override;

protected:
    scalar m_weight;
};

END_NAMESPACE()
