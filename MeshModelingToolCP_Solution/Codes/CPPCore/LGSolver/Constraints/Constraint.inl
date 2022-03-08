#pragma once

#include "Constraint.h"

BEGIN_NAMESPACE(AAShapeUp)

//// Begin ConstraintAbstract

template<i32 Dim>
inline ConstraintAbstract<Dim>::ConstraintAbstract(const std::vector<i32>& idIncidentPoints, scalar weight)
    : m_idIncidentPoints(Eigen::Map<const VectorXi>(idIncidentPoints.data(), idIncidentPoints.size()))
    , m_weight(weight)
    , m_idConstraint(INVALID_INT)
{
    assert(m_idIncidentPoints.size() > 0);
}

template<i32 Dim>
inline ConstraintAbstract<Dim>::~ConstraintAbstract()
{
}

template<i32 Dim>
inline void ConstraintAbstract<Dim>::setIdConstraint(i32 idConstraint)
{
    m_idConstraint = idConstraint;
}

template<i32 Dim>
inline i32 ConstraintAbstract<Dim>::getIdConstraint() const
{
    return m_idConstraint;
}

template<i32 Dim>
inline const VectorXi& ConstraintAbstract<Dim>::getIdIncidentPoints() const
{
    return m_idIncidentPoints;
}

template<i32 Dim>
inline scalar ConstraintAbstract<Dim>::getWeight() const
{
    return m_weight;
}

template<i32 Dim>
inline i32 ConstraintAbstract<Dim>::numIndices() const
{
    return m_idIncidentPoints.size(); 
}

//// End ConstraintAbstract

//// Begin ConstraintBase

template<i32 Dim, typename TConstraintProjectionOperator, typename TConstraintTripletGenerator, typename TInvariantTransformer>
inline ConstraintBase<Dim, TConstraintProjectionOperator, TConstraintTripletGenerator, TInvariantTransformer>::ConstraintBase(
    const std::vector<i32>& idIncidentPoints, scalar weight)
    : ConstraintAbstract<Dim>(idIncidentPoints, weight)
{
    this->m_invariantTransformer.m_transformedPoints.resize(Dim, m_invariantTransformer.numTransformedPointsToCreate(static_cast<i32>(this->m_idIncidentPoints.size())));
}

template<i32 Dim, typename TConstraintProjectionOperator, typename TConstraintTripletGenerator, typename TInvariantTransformer>
inline ConstraintBase<Dim, TConstraintProjectionOperator, TConstraintTripletGenerator, TInvariantTransformer>::~ConstraintBase()
{
}

template<i32 Dim, typename TConstraintProjectionOperator, typename TConstraintTripletGenerator, typename TInvariantTransformer>
inline i32 ConstraintBase<Dim, TConstraintProjectionOperator, TConstraintTripletGenerator, TInvariantTransformer>::numTransformedPoints() const
{
    return static_cast<i32>(m_invariantTransformer.m_transformedPoints.cols());
}

template<i32 Dim, typename TConstraintProjectionOperator, typename TConstraintTripletGenerator, typename TInvariantTransformer>
inline scalar ConstraintBase<Dim, TConstraintProjectionOperator, TConstraintTripletGenerator, TInvariantTransformer>::project(
    const MatrixNX& points, MatrixNX& projections)
{
    this->m_invariantTransformer.generateTransformPoints(*this, points);

    scalar weightedDist = this->m_projectionOperator.project(*this, this->m_invariantTransformer.m_transformedPoints, projections);

    return weightedDist;
}

template<i32 Dim, typename TConstraintProjectionOperator, typename TConstraintTripletGenerator, typename TInvariantTransformer>
inline void ConstraintBase<Dim, TConstraintProjectionOperator, TConstraintTripletGenerator, TInvariantTransformer>::extractConstraint(
    std::vector<SMatrixTriplet>& triplets, i32& inOutConstraintId)
{
    i64 nIdx = this->m_idIncidentPoints.size();

    // Set constraint ID, for projection.
    this->setIdConstraint(inOutConstraintId);

    this->m_tripletGenerator.generateTriplets(*this, triplets, inOutConstraintId);
}

template<i32 Dim, typename TConstraintProjectionOperator, typename TConstraintTripletGenerator, typename TInvariantTransformer>
inline void ConstraintBase<Dim, TConstraintProjectionOperator, TConstraintTripletGenerator, TInvariantTransformer>::generateTransformPoints(const MatrixNX& points)
{
    static_assert(std::is_base_of_v<InvariantTransformerAbstract<Dim, ConstraintAbstract<Dim> >, TInvariantTransformer>);

    this->m_invariantTransformer.generateTransformPoints(*this, points);
}

//// End ConstraintBase

//// Begin IConstraintComponent

template<i32 Dim, typename TConstraintAbstract>
inline constexpr IConstraintComponent<Dim, TConstraintAbstract>::IConstraintComponent()
{
    // The TConstraintAbstract can only be ConstraintAbstract<Dim>
    static_assert(std::is_same_v<ConstraintAbstract<TConstraintAbstract::getDim()>, TConstraintAbstract>);
}

template<i32 Dim, typename TConstraintAbstract>
template<typename TConstraint>
inline constexpr void IConstraintComponent<Dim, TConstraintAbstract>::staticTypeCheckBase() const
{
    static_assert(std::is_base_of_v<ConstraintAbstract<TConstraint::getDim()>, TConstraint>);
    static_assert(!std::is_same_v<ConstraintAbstract<TConstraint::getDim()>, TConstraint>);
}

template<i32 Dim, typename TConstraintAbstract>
template<typename TConstraint>
inline constexpr void IConstraintComponent<Dim, TConstraintAbstract>::staticConvertibleCheckBase() const
{
    static_assert(std::is_convertible_v<TConstraint, ConstraintAbstract<TConstraint::getDim()> >);
}

template<i32 Dim, typename TConstraintAbstract>
template<typename TConstraint>
inline constexpr TConstraint& IConstraintComponent<Dim, TConstraintAbstract>::staticCast(TConstraintAbstract& constraint) const
{
    this->staticConvertibleCheckBase<TConstraint>();
    return static_cast<TConstraint&>(constraint);
}

template<i32 Dim, typename TConstraintAbstract>
template<typename TConstraint>
inline constexpr TConstraint* IConstraintComponent<Dim, TConstraintAbstract>::staticCast(TConstraintAbstract* constraint) const
{
    this->staticConvertibleCheckBase<TConstraint>();
    return static_cast<TConstraint*>(constraint);
}

//// End IConstraintComponent

//// Begin IdentityTransformer

template<i32 Dim, typename TConstraintAbstract>
inline constexpr i32 IdentityTransformer<Dim, TConstraintAbstract>::numTransformedPointsToCreate(const i32 numIndices) const
{
    return numIndices;
}

template<i32 Dim, typename TConstraintAbstract>
inline void IdentityTransformer<Dim, TConstraintAbstract>::generateTransformPoints(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& points)
{
    for (i64 i = 0; i < constraint.getIdIncidentPoints().size(); ++i)
    {
        this->m_transformedPoints.col(i) = points.col(constraint.getIdIncidentPoints()[i]);
    }
}

//// End IdentityTransformer

//// Begin MeanCenteringTransformer

template<i32 Dim, typename ConstraintAbstract>
inline constexpr i32 MeanCenteringTransformer<Dim, ConstraintAbstract>::numTransformedPointsToCreate(const i32 numIndices) const
{
    return numIndices;
}

template<i32 Dim, typename TConstraintAbstract>
inline void MeanCenteringTransformer<Dim, TConstraintAbstract>::generateTransformPoints(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& points)
{
    for (i64 i = 0; i < constraint.getIdIncidentPoints().size(); ++i)
    {
        this->m_transformedPoints.col(i) = points.col(constraint.getIdIncidentPoints()[i]);
    }
    typename TConstraintAbstract::VectorN meanPoint = constraint.m_transformedPoints.rowwise().mean();
    constraint.m_transformedPoints.colwise() -= meanPoint;
}

//// End MeanCenteringTransformer

//// Begin SubtractFirstTransformer

template<i32 Dim, typename TConstraintAbstract>
inline constexpr i32 SubtractFirstTransformer<Dim, TConstraintAbstract>::numTransformedPointsToCreate(const i32 numIndices) const
{
    return numIndices - 1;
}

template<i32 Dim, typename TConstraintAbstract>
inline void SubtractFirstTransformer<Dim, TConstraintAbstract>::generateTransformPoints(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& points)
{
    typename TConstraintAbstract::VectorN firstPoint = points.col(constraint.getIdIncidentPoints()[0]);
    for (i64 i = 1; i < constraint.getIdIncidentPoints().size(); ++i)
    {
        this->m_transformedPoints.col(i - 1) = points.col(constraint.getIdIncidentPoints()[i]) - firstPoint;
    }
}

//// End MeanCenteringTransformer

//// Begin IdentityWeightTripletGenerator

template<i32 Dim, typename TConstraintAbstract>
inline void IdentityWeightTripletGenerator<Dim, TConstraintAbstract>::generateTriplets(TConstraintAbstract& constraint, std::vector<SMatrixTriplet>& triplets, i32& inOutConstraintId) const
{
    i64 nIdx = constraint.getIdIncidentPoints().size();
    for (i64 i = 0; i < nIdx; ++i)
    {
        triplets.push_back(SMatrixTriplet(inOutConstraintId, constraint.getIdIncidentPoints()[i], constraint.getWeight()));
        ++inOutConstraintId;
    }
}

//// End IdentityWeightTripletGenerator

//// Begin MeanCenteringWeightTripletGenerator

template<i32 Dim, typename TConstraintAbstract>
inline void MeanCenteringWeightTripletGenerator<Dim, TConstraintAbstract>::generateTriplets(TConstraintAbstract& constraint, std::vector<SMatrixTriplet>& triplets, i32& inOutConstraintId) const
{
    i64 nIdx = constraint.getIdIncidentPoints().size();
    scalar coefDiff = -constraint.getWeight() / nIdx;
    scalar coefCenter = constraint.getWeight() + coefDiff;
    for (i64 i = 0; i < nIdx; ++i)
    {
        for (i64 j = 0; j < nIdx; ++j)
        {
            triplets.push_back(SMatrixTriplet(inOutConstraintId, constraint.getIdIncidentPoints()[0], i == j ? coefCenter : coefDiff));
        }
        ++inOutConstraintId;
    }
}

//// End MeanCenteringWeightTripletGenerator

//// Begin SubtractFirstWeightTripletGenerator

template<i32 Dim, typename TConstraintAbstract>
inline void SubtractFirstWeightTripletGenerator<Dim, TConstraintAbstract>::generateTriplets(TConstraintAbstract& constraint, std::vector<SMatrixTriplet>& triplets, i32& inOutConstraintId) const
{
    i64 nIdx = constraint.getIdIncidentPoints().size();
    for (i64 i = 1; i < nIdx; ++i)
    {
        triplets.push_back(SMatrixTriplet(inOutConstraintId, constraint.getIdIncidentPoints()[0], -constraint.getWeight()));
        triplets.push_back(SMatrixTriplet(inOutConstraintId, constraint.getIdIncidentPoints()[i], constraint.getWeight()));
        ++inOutConstraintId;
    }
}

//// End SubtractFirstWeightTripletGenerator

//// Begin IdentityProjectionOperator

template<i32 Dim, typename TConstraintAbstract>
inline scalar IdentityProjectionOperator<Dim, TConstraintAbstract>::project(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& transformedPoints, typename TConstraintAbstract::MatrixNX& projections) const
{
    using MatrixNX = typename TConstraintAbstract::MatrixNX;

    Eigen::Map<MatrixNX> projectionBlock(&projections(0, constraint.getIdConstraint()), Dim, transformedPoints.cols());

    projectionBlock = transformedPoints;

    scalar sqrDist = (transformedPoints - projectionBlock).squaredNorm();
    
    projectionBlock *= constraint.getWeight();

    return sqrDist * (constraint.getWeight() * constraint.getWeight()) * static_cast<scalar>(0.5);
}

//// End IdentityProjectionOperator

//// Begin ConstraintSet

template<i32 Dim>
inline i32 ConstraintSetAbstract<Dim>::addConstraint(const std::shared_ptr<ConstraintAbstract<Dim>>& constraintShPtr)
{
    m_constraintShPtrs.push_back(constraintShPtr);
    return static_cast<i32>(m_constraintShPtrs.size());
}

template<i32 Dim>
inline const std::vector<std::shared_ptr<ConstraintAbstract<Dim>>>& ConstraintSetAbstract<Dim>::getConstraints() const
{
    return m_constraintShPtrs;
}

template<i32 Dim>
inline void ConstraintSetAbstract<Dim>::clearConstraints()
{
    m_constraintShPtrs.clear();
}

//// End ConstraintSet

END_NAMESPACE()
