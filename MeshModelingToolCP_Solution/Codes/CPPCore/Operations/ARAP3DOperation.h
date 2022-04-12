#pragma once

#include "OperationBase.h"

BEGIN_NAMESPACE(AAShapeUp)

class ARAP3DOperation : public OperationBase
{
public:
    using Super = OperationBase;

    ARAP3DOperation(const std::shared_ptr<SolverAbstract<3, LinearRegularizer<3>, ConstraintSet<3>>>& solverShPtr)
        : Super(solverShPtr) {}

    virtual ~ARAP3DOperation() {}

    virtual bool initializeConstraintsAndRegularizations() override;

    virtual std::tuple<MeshDirtyFlag, MeshIndexType> getOutputErrors(std::vector<scalar>& outErrors) const override;

    virtual MeshDirtyFlag getMeshDirtyFlag() const override;

    bool solve(Matrix3X& newPositions, i32 nIter) override;


    //EigenMesh<3> refMesh;
    //scalar closeness_weight = scalar(1), relative_laplacian_weight = scalar(0.1), laplacian_weight = scalar(0.1), planarity_weight = scalar(1);

protected:
};

END_NAMESPACE()
