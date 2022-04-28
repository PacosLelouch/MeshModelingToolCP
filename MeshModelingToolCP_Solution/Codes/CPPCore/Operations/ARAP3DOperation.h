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

    void markUsingCache() { m_usingCache = true; }

    scalar m_deformationWeight = scalar(1);
protected:
    std::shared_ptr<tetgenio> m_tempTetMeshIOShPtr;
    std::shared_ptr<EigenMesh<3> > m_tempTetEigenMeshShPtr;
    bool m_usingCache = false;
};

END_NAMESPACE()
