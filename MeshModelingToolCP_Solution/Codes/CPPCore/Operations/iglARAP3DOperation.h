#pragma once

#include "OperationBase.h"
#include <igl/arap.h>

BEGIN_NAMESPACE(AAShapeUp)

class iglARAP3DOperation : public OperationBase
{
public:
    using Super = OperationBase;

    iglARAP3DOperation(const std::shared_ptr<SolverAbstract<3, LinearRegularizer<3>, ConstraintSet<3>>>& solverShPtr)
        : Super(solverShPtr) {}

    virtual ~iglARAP3DOperation() {}

    virtual bool initializeConstraintsAndRegularizations() override;

    virtual std::tuple<MeshDirtyFlag, MeshIndexType> getOutputErrors(std::vector<scalar>& outErrors) const override;

    virtual MeshDirtyFlag getMeshDirtyFlag() const override;

    bool solve(Matrix3X& newPositions, i32 nIter) override;

    scalar m_deformationWeight = scalar(1);

    igl::ARAPData arapData;
    VectorXi lastHandleIndices;
    Matrix3X originVerts;
    Matrix3Xi faceIndices;
protected:
};

END_NAMESPACE()
