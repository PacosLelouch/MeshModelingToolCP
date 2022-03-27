#pragma once

#include "OperationBase.h"

BEGIN_NAMESPACE(AAShapeUp)

class TestBoundingSphereOperation : public OperationBase
{
public:
    using Super = OperationBase;

    TestBoundingSphereOperation(const std::shared_ptr<SolverAbstract<3, LinearRegularizer<3>, ConstraintSet<3>>>& solverShPtr, scalar sphereProjectionWeight = scalar(1), scalar LaplacianWeight = scalar(1))
        : Super(solverShPtr)
        , m_sphereProjectionWeight(sphereProjectionWeight)
        , m_LaplacianWeight(LaplacianWeight){}

    virtual ~TestBoundingSphereOperation() {}

    virtual bool initializeConstraintsAndRegularizations() override;

    virtual std::tuple<MeshDirtyFlag, MeshIndexType> getOutputErrors(std::vector<scalar>& outErrors) const override;

    virtual MeshDirtyFlag getMeshDirtyFlag() const override;

    scalar m_sphereProjectionWeight = scalar(1);
    scalar m_LaplacianWeight = scalar(0.1);

protected:
    Vector3 m_center, m_maxCorner;
    scalar m_radius = scalar(0);
};

END_NAMESPACE()
