#pragma once

#include "OperationBase.h"

BEGIN_NAMESPACE(igl)
struct ARAPData;
END_NAMESPACE()

#define TET_GEN_FOR_IGL_ARAP3D 1

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

    void markUsingCache() { m_usingCache = true; }

    //bool useTetGen = true;

    scalar m_deformationWeight = scalar(1);

protected:
    bool m_usingCache = false;
    std::shared_ptr<tetgenio> m_tempTetMeshIOShPtr;
    std::shared_ptr<EigenMesh<3> > m_tempTetEigenMeshShPtr;

    std::shared_ptr<struct igl::ARAPData> arapDataShPtr;
    std::vector<i32> lastHandleIndices;
    Matrix3X originVerts;
#if !TET_GEN_FOR_IGL_ARAP3D
    Matrix3Xi faceIndices; // Without TetGen
#else // TET_GEN_FOR_IGL_ARAP3D
    Matrix4Xi tetIndices; // With TetGen
#endif // TET_GEN_FOR_IGL_ARAP3D
};

END_NAMESPACE()
