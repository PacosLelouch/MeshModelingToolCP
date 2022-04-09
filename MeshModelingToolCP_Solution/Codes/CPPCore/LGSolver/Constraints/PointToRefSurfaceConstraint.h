#pragma once

#include "Constraint.h"
#include "MeshAABB.h"
#include "EigenMesh.h"

BEGIN_NAMESPACE(AAShapeUp)

class PointToRefSurfaceProjectionOperator : public ConstraintProjectionOperatorAbstract<3, ConstraintAbstract<3>>
{
public:
    scalar project(ConstraintAbstract<3>& constraint, const typename ConstraintAbstract<3>::MatrixNX& transformedPoints, typename ConstraintAbstract<3>::MatrixNX& projections) const;

    std::shared_ptr<MeshAABB> refMeshTree;
};

class PointToRefSurfaceConstraint : public ConstraintBase<3,
    PointToRefSurfaceProjectionOperator,
    IdentityWeightTripletGenerator<3>,
    IdentityTransformer<3>>
{
public:
    using Super = ConstraintBase<3,
        PointToRefSurfaceProjectionOperator,
        IdentityWeightTripletGenerator<3>,
        IdentityTransformer<3> >;

    PointToRefSurfaceConstraint(i32 idx, scalar weight, std::shared_ptr<MeshAABB> refMeshTree)
        : Super(std::vector<i32>({ idx }), weight)
    {
        this->m_projectionOperator.refMeshTree = refMeshTree; //NOTICE: the refMesh should be triangualted
    }

    virtual ~PointToRefSurfaceConstraint() {}
};

inline scalar PointToRefSurfaceProjectionOperator::project(ConstraintAbstract<3>& constraint, const typename ConstraintAbstract<3>::MatrixNX& transformedPoints, typename ConstraintAbstract<3>::MatrixNX& projections) const
{
    using MatrixNX = typename ConstraintAbstract<3>::MatrixNX;
    Eigen::Map<MatrixNX> projectionBlock(&projections(0, constraint.getIdConstraint()), 3, transformedPoints.cols());

    Vector3 closestPoint;
    refMeshTree->getClosestPoint(transformedPoints.col(0), closestPoint);
    projectionBlock.col(0) = closestPoint;

    //general code for projection and error
    scalar sqrDist = (transformedPoints - projectionBlock).squaredNorm();
    projectionBlock *= constraint.getWeight();
    return sqrDist * (constraint.getWeight()) * static_cast<scalar>(0.5);
    //return sqrDist * (constraint.getWeight() * constraint.getWeight()) * static_cast<scalar>(0.5);
}

END_NAMESPACE()
