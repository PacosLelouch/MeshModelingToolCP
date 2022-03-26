#pragma once

#include "Constraint.h"

BEGIN_NAMESPACE(AAShapeUp)


class PlaneProjectionOperator : public ConstraintProjectionOperatorAbstract<3, ConstraintAbstract<3>>
{
public:
    scalar project(ConstraintAbstract<3>& constraint, const typename ConstraintAbstract<3>::MatrixNX& transformedPoints, typename ConstraintAbstract<3>::MatrixNX& projections) const;
};

class PlaneConstraint : public ConstraintBase<3,
    PlaneProjectionOperator,
    MeanCenteringWeightTripletGenerator<3>,
    MeanCenteringTransformer<3>>
{
public:
    using Super = ConstraintBase<3,
        PlaneProjectionOperator,
        MeanCenteringWeightTripletGenerator<3>,
        MeanCenteringTransformer<3> >;

    PlaneConstraint(const std::vector<int>& idI, scalar weight)
        : Super(idI, weight)
    {
    }

    virtual ~PlaneConstraint() {}
};

inline scalar PlaneProjectionOperator::project(ConstraintAbstract<3>& constraint, const typename ConstraintAbstract<3>::MatrixNX& transformedPoints, typename ConstraintAbstract<3>::MatrixNX& projections) const
{
    using MatrixNX = typename ConstraintAbstract<3>::MatrixNX;

    Eigen::Map<MatrixNX> projectionBlock(&projections(0, constraint.getIdConstraint()), 3, transformedPoints.cols());

    Eigen::JacobiSVD<Matrix3X> jSVD;
    jSVD.compute(transformedPoints, Eigen::ComputeFullU);
    Vector3 bestFitNormal = jSVD.matrixU().col(2).normalized();
    projectionBlock = transformedPoints
        - bestFitNormal * (bestFitNormal.transpose() * transformedPoints);

    scalar sqrDist = (transformedPoints - projectionBlock).squaredNorm();
    return sqrDist * (constraint.getWeight() * constraint.getWeight()) * static_cast<scalar>(0.5);
}

END_NAMESPACE()
