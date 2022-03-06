#pragma once

#include "Constraint.h"

BEGIN_NAMESPACE(AAShapeUp)

//// Begin ConstraintAbstract

template<i32 N>
inline ConstraintAbstract<N>::ConstraintAbstract(const std::vector<i32>& idIncidentPoints, scalar weight)
    : m_idIncidentPoints(Eigen::Map<const VectorXi>(idIncidentPoints.data(), idIncidentPoints.size()))
    , m_weight(weight)
    , m_idConstraint(INVALID_INT)
{
    assert(m_idIncidentPoints.size() > 0);
}

template<i32 N>
inline ConstraintAbstract<N>::~ConstraintAbstract()
{
}

template<i32 N>
inline void ConstraintAbstract<N>::setIdConstraint(i32 idConstraint)
{
    m_idConstraint = idConstraint;
}

template<i32 N>
inline i32 ConstraintAbstract<N>::numIndices() const
{
    return m_idIncidentPoints.size(); 
}

template<i32 N>
inline i32 ConstraintAbstract<N>::numTransformedPoints() const
{
    return m_transformedPoints.cols();
}

//// End ConstraintAbstract

//// Begin ConstraintBase

template<i32 N, typename TInvariantTransformer, typename TConstraintTripletGenerator, typename TConstraintProjectionOperator>
inline ConstraintBase<N, TInvariantTransformer, TConstraintTripletGenerator, TConstraintProjectionOperator>::ConstraintBase(const std::vector<i32>& idIncidentPoints, scalar weight)
    : ConstraintAbstract<N>(idIncidentPoints, weight)
{
    this->m_transformedPoints.resize(N, invariantTransformer.numTransformedPointsToCreate(static_cast<i32>(this->m_idIncidentPoints.size())));
}

template<i32 N, typename TInvariantTransformer, typename TConstraintTripletGenerator, typename TConstraintProjectionOperator>
inline ConstraintBase<N, TInvariantTransformer, TConstraintTripletGenerator, TConstraintProjectionOperator>::~ConstraintBase()
{
}

template<i32 N, typename TInvariantTransformer, typename TConstraintTripletGenerator, typename TConstraintProjectionOperator>
inline scalar ConstraintBase<N, TInvariantTransformer, TConstraintTripletGenerator, TConstraintProjectionOperator>::project(const MatrixNX& points, MatrixNX& projections)
{
    generateTransformPoints(points);

    Eigen::Map<MatrixNX> projectionBlock = Eigen::Map<MatrixNX>(&projections(0, this->m_idConstraint), N, this->m_transformedPoints.cols());

    //TODO

    return scalar();
}

template<i32 N, typename TInvariantTransformer, typename TConstraintTripletGenerator, typename TConstraintProjectionOperator>
inline i32 ConstraintBase<N, TInvariantTransformer, TConstraintTripletGenerator, TConstraintProjectionOperator>::extractConstraint(std::vector<SMatrixTriplet>& triplets)
{
    i64 nIdx = this->m_idIncidentPoints.size();

    //TODO

    return this->m_idConstraint;
}

template<i32 N, typename TInvariantTransformer, typename TConstraintTripletGenerator, typename TConstraintProjectionOperator>
inline void ConstraintBase<N, TInvariantTransformer, TConstraintTripletGenerator, TConstraintProjectionOperator>::generateTransformPoints(const MatrixNX& points)
{
    static_assert(std::is_base_of_v<IInvariantTransformer<N, ConstraintAbstract<N> >, TInvariantTransformer>);

    this->invariantTransformer.generateTransformPoints(*this, points);
}

//// End ConstraintBase

//// Begin IConstraintComponent

template<i32 N, typename TConstraintAbstract>
inline constexpr IConstraintComponent<N, TConstraintAbstract>::IConstraintComponent()
{
    static_assert(std::is_same_v<ConstraintAbstract<TConstraintAbstract::getDim()>, TConstraintAbstract>);
}

template<i32 N, typename TConstraintAbstract>
template<typename TConstraint>
inline constexpr void IConstraintComponent<N, TConstraintAbstract>::staticCheckBase() const
{
    static_assert(std::is_base_of_v<ConstraintAbstract<TConstraint::getDim()>, TConstraint>);
}

//// End IConstraintComponent

//// Begin IdentityTransformer

template<i32 N, typename TConstraintAbstract>
inline constexpr i32 IdentityTransformer<N, TConstraintAbstract>::numTransformedPointsToCreate(const i32 numIndices) const
{
    return numIndices;
}

template<i32 N, typename TConstraintAbstract>
template<typename TConstraint>
inline void IdentityTransformer<N, TConstraintAbstract>::generateTransformPoints(TConstraint& constraint, const typename TConstraintAbstract::MatrixNX& points) const
{
    this->staticCheckBase<TConstraint>();
    //static_assert(std::is_base_of_v<ConstraintAbstract<TConstraint::getDim()>, TConstraint>);

    for (i64 i = 0; i < constraint.m_idIncidentPoints.size(); ++i)
    {
        constraint.m_transformedPoints.col(i) = points.col(constraint.m_idIncidentPoints[i]);
    }
}

//// End IdentityTransformer

//// Begin MeanCenteringTransformer

template<i32 N, typename ConstraintAbstract>
inline constexpr i32 MeanCenteringTransformer<N, ConstraintAbstract>::numTransformedPointsToCreate(const i32 numIndices) const
{
    return numIndices;
}

template<i32 N, typename TConstraintAbstract>
template<typename TConstraint>
inline void MeanCenteringTransformer<N, TConstraintAbstract>::generateTransformPoints(TConstraint& constraint, const typename TConstraintAbstract::MatrixNX& points) const
{
    this->staticCheckBase<TConstraint>();
    //static_assert(std::is_base_of_v<ConstraintAbstract<TConstraint::getDim()>, TConstraint>);

    for (i64 i = 0; i < constraint.m_idIncidentPoints.size(); ++i)
    {
        constraint.m_transformedPoints.col(i) = points.col(constraint.m_idIncidentPoints[i]);
    }
    typename TConstraintAbstract::VectorN meanPoint = constraint.m_transformedPoints.rowwise().mean();
    constraint.m_transformedPoints.colwise() -= meanPoint;
}

//// End MeanCenteringTransformer

//// Begin SubtractFirstTransformer

template<i32 N, typename TConstraintAbstract>
inline constexpr i32 SubtractFirstTransformer<N, TConstraintAbstract>::numTransformedPointsToCreate(const i32 numIndices) const
{
    return numIndices - 1;
}

template<i32 N, typename TConstraintAbstract>
template<typename TConstraint>
inline void SubtractFirstTransformer<N, TConstraintAbstract>::generateTransformPoints(TConstraint& constraint, const typename TConstraintAbstract::MatrixNX& points) const
{
    this->staticCheckBase<TConstraint>();
    //static_assert(std::is_base_of_v<ConstraintAbstract<TConstraint::getDim()>, TConstraint>);

    typename TConstraintAbstract::VectorN firstPoint = constraint.points.cols(constraint.m_idIncidentPoints[0]);
    for (i64 i = 1; i < constraint.m_idIncidentPoints.size(); ++i)
    {
        constraint.m_transformedPoints.col(i - 1) = points.col(constraint.m_idIncidentPoints[i]) - firstPoint;
    }
}

//// End MeanCenteringTransformer

END_NAMESPACE()
