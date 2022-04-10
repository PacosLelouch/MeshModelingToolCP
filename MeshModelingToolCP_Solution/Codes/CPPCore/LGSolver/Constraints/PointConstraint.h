#pragma once

#include "Constraint.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class PointProjectionOperator : public ConstraintProjectionOperatorAbstract<Dim, TConstraintAbstract>
{
public:
    using Super = ConstraintProjectionOperatorAbstract<Dim, TConstraintAbstract>;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)
public:
    scalar project(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& transformedPoints, typename TConstraintAbstract::MatrixNX& projections) const;

    VectorN m_targetPosition;
};

template<i32 Dim>
class PointConstraint : public ConstraintBase<Dim,
    PointProjectionOperator<Dim>,
    IdentityWeightTripletGenerator<Dim>,
    IdentityTransformer<Dim> >
{
public:
    using Super = ConstraintBase<Dim,
        PointProjectionOperator<Dim>,
        IdentityWeightTripletGenerator<Dim>,
        IdentityTransformer<Dim> >;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)
public:

    PointConstraint(i32 idx, scalar weight, VectorN targetPosition)
        : Super(std::vector<i32>({idx}), weight)
    {
        this->m_projectionOperator.m_targetPosition = targetPosition;
    }

    virtual ~PointConstraint() {}
};

template<i32 Dim, typename TConstraintAbstract>
inline scalar PointProjectionOperator<Dim, TConstraintAbstract>::project(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& transformedPoints, typename TConstraintAbstract::MatrixNX& projections) const
{
    Eigen::Map<MatrixNX> projectionBlock(&projections(0, constraint.getIdConstraint()), Dim, transformedPoints.cols());

    projectionBlock.col(0) = this->m_targetPosition;

    //general code for projection and error
    scalar sqrDist = (transformedPoints - projectionBlock).squaredNorm();

    // Don't forget it!
    scalar sqrtWeight = constraint.getSqrtWeight();//glm::sqrt(constraint.getWeight());
    projectionBlock *= sqrtWeight;

    return sqrDist * (constraint.getWeight()) * static_cast<scalar>(0.5);
    //return sqrDist * (constraint.getWeight() * constraint.getWeight()) * static_cast<scalar>(0.5);
}

using PointConstraint2D = PointConstraint<2>;
using PointConstraint3D = PointConstraint<3>;

END_NAMESPACE()
