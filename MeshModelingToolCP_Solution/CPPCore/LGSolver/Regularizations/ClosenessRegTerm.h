#pragma once

#include "RegularizationTerm.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
class ClosenessRegTerm : public RegularizationTermAbstract<Dim>
{
public:
    using Super = RegularizationTermAbstract<Dim>;

    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    ClosenessRegTerm(i32 idx, scalar weight, const VectorN& targetPoint);

    virtual void evaluate(VectorXi& outPointIndices, VectorX& outCoefficients, VectorN& outValue) const override;

protected:
    i32 m_idx;
    scalar m_weight;
    VectorN m_targetPoint;
};

template<i32 Dim>
inline ClosenessRegTerm<Dim>::ClosenessRegTerm(i32 idx, scalar weight, const VectorN& targetPoint)
    : m_idx(idx)
    , m_weight(weight)
    , m_targetPoint(targetPoint)
{
}

template<i32 Dim>
inline void ClosenessRegTerm<Dim>::evaluate(VectorXi& outPointIndices, VectorX& outCoefficients, VectorN& outValue) const
{
    scalar sqrtWeight(glm::sqrt(weight));

    outPointIndices.resize(1);
    outPointIndices(0) = m_idx;

    outCoefficients.resize(1);
    outCoefficients(0) = sqrtWeight;

    outValue = m_targetPoint * sqrtWeight;
}


END_NAMESPACE()