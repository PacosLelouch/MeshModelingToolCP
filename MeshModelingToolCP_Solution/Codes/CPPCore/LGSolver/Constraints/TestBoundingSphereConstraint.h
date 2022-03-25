#pragma once

#include "Constraint.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class TestBoundingSphereProjectionOperator : public ConstraintProjectionOperatorAbstract<Dim, TConstraintAbstract>
{
public:
    using Super = ConstraintProjectionOperatorAbstract<Dim, TConstraintAbstract>;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)
public:
    scalar project(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& transformedPoints, typename TConstraintAbstract::MatrixNX& projections) const;

    VectorN m_center;
    scalar m_radius;
};

template<i32 Dim>
class TestBoundingSphereConstraint : public ConstraintBase<Dim,
    TestBoundingSphereProjectionOperator<Dim>,
    IdentityWeightTripletGenerator<Dim>,
    IdentityTransformer<Dim> >
{
public:
    using Super = ConstraintBase<Dim,
        TestBoundingSphereProjectionOperator<Dim>,
        IdentityWeightTripletGenerator<Dim>,
        IdentityTransformer<Dim> >;
    USING_MATRIX_VECTOR_SHORTNAME(Dim)
    //USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)

    TestBoundingSphereConstraint(i32 pointIdx, scalar weight, VectorN center, scalar radius)
        : Super(std::vector<i32>({pointIdx}), weight)
    {
        this->m_projectionOperator.m_center = center;
        this->m_projectionOperator.m_radius = radius;
    }

    virtual ~TestBoundingSphereConstraint() {}
};

template<i32 Dim, typename TConstraintAbstract>
inline scalar TestBoundingSphereProjectionOperator<Dim, TConstraintAbstract>::project(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& transformedPoints, typename TConstraintAbstract::MatrixNX& projections) const
{
    Eigen::Map<MatrixNX> projectionBlock(&projections(0, constraint.getIdConstraint()), Dim, transformedPoints.cols());

    VectorN diff = transformedPoints.col(0) - this->m_center;

    projectionBlock.col(0) = diff.normalized() * this->m_radius;

    scalar sqrDist = (transformedPoints - projectionBlock).squaredNorm();

    projectionBlock *= constraint.getWeight();

    return sqrDist * (constraint.getWeight() * constraint.getWeight()) * static_cast<scalar>(0.5);
}

using TestBoundingSphereConstraint2D = TestBoundingSphereConstraint<2>;
using TestBoundingSphereConstraint3D = TestBoundingSphereConstraint<3>;

END_NAMESPACE()
