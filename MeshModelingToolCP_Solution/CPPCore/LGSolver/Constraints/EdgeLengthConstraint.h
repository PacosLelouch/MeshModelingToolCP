#pragma once

#include "Constraint.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class EdgeLengthProjectionOperator : public ConstraintProjectionOperatorAbstract<Dim, TConstraintAbstract>
{
public:
    scalar project(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& transformedPoints, typename TConstraintAbstract::MatrixNX& projections) const;

    scalar m_targetLength;
};

template<i32 Dim>
class EdgeLengthConstraint : public ConstraintBase<Dim,
    EdgeLengthProjectionOperator<Dim>,
    SubtractFirstWeightTripleGenerator<Dim>,
    SubtractFirstTransformer<Dim> >
{
public:
    using Super = ConstraintBase<Dim,
        EdgeLengthProjectionOperator<Dim>,
        SubtractFirstWeightTripleGenerator<Dim>,
        SubtractFirstTransformer<Dim> >;

    EdgeLengthConstraint(i32 idx1, i32 idx2, scalar weight, scalar targetLength)
        : Super(std::vector<i32>({idx1, idx2}), weight)
    {
        this->m_projectionOperator.m_targetLength = targetLength;
    }

    virtual ~EdgeLengthConstraint() {}
};

template<i32 Dim, typename TConstraintAbstract>
inline scalar EdgeLengthProjectionOperator<Dim, TConstraintAbstract>::project(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& transformedPoints, typename TConstraintAbstract::MatrixNX& projections) const
{
    using MatrixNX = typename TConstraintAbstract::MatrixNX;

    Eigen::Map<MatrixNX> projectionBlock(&projections(0, constraint.getIdConstraint()), Dim, transformedPoints.cols());

    projectionBlock.col(0) = transformedPoints.col(0).normalized() * this->m_targetLength;

    scalar sqrDist = (transformedPoints - projectionBlock).squaredNorm();

    projectionBlock *= constraint.getWeight();

    return sqrDist * (constraint.getWeight() * constraint.getWeight()) * static_cast<scalar>(0.5);
}

using EdgeLengthConstraint2D = EdgeLengthConstraint<2>;
using EdgeLengthConstraint3D = EdgeLengthConstraint<3>;

END_NAMESPACE()
