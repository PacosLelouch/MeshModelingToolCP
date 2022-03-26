#pragma once

#include "LaplacianRegTerm.h"

BEGIN_NAMESPACE(AAShapeUp)

//// Begin LaplacianRegTermBase

template<i32 Dim>
inline LaplacianRegTermBase<Dim>::LaplacianRegTermBase(const std::vector<i32>& indices, const std::vector<scalar>& coefs, scalar weight, const MatrixNX* refPointsPtr)
    : m_pointIndices(indices)
    , m_coefficients(coefs)
    , m_weight(weight)
    , m_refPointsPtr(refPointsPtr)
{
}

template<i32 Dim>
inline void LaplacianRegTermBase<Dim>::evaluate(VectorXi& outPointIndices, VectorX& outCoefficients, VectorN& outValue) const
{
    this->generateLaplacianHelper(outPointIndices, outCoefficients, outValue, this->m_pointIndices, this->m_coefficients, this->m_weight, this->m_refPointsPtr);
}

template<i32 Dim>
inline void LaplacianRegTermBase<Dim>::generateLaplacianHelper(
    VectorXi& outPointIndices, VectorX& outCoefficients, VectorN& outValue,
    const std::vector<int>& indices, const std::vector<scalar>& coefs, scalar weight, const MatrixNX* refPoints) const
{
    assert(indices.size() == coefs.size());

    scalar sqrtWeight(glm::sqrt(weight));

    outPointIndices = Eigen::Map<const VectorXi>(indices.data(), indices.size());
    
    outCoefficients = Eigen::Map<const VectorX>(coefs.data(), coefs.size()) * sqrtWeight;

    outValue = VectorN::Zero();
    if (refPoints)
    {
        for (i64 i = 0; i < i64(indices.size()); ++i)
        {
            outValue += refPoints->col(indices[i]) * coefs[i];
        }
    }
}

//// End LaplacianRegTermBase

//// Begin UniformLaplacianRegTerm

template<i32 Dim>
inline UniformLaplacianRegTerm<Dim>::UniformLaplacianRegTerm(const std::vector<i32>& indices, scalar weight)
    : Super(indices, {}, weight, nullptr)
{
    i32 nPoints = static_cast<i32>(indices.size());
    this->m_coefficients.reserve(nPoints);
    this->m_coefficients.push_back(scalar(1.0));
    this->m_coefficients.insert(this->m_coefficients.end(), nPoints - 1, scalar(-1.0 / (nPoints - 1)));
}

//// End UniformLaplacianRegTerm

//// Begin LaplacianRegTerm

template<i32 Dim>
inline LaplacianRegTerm<Dim>::LaplacianRegTerm(const std::vector<i32>& indices, const std::vector<scalar>& coefs, scalar weight)
    : Super(indices, coefs, weight, nullptr)
{
}

//// End LaplacianRegTerm

//// Begin UniformLaplacianRelativeRegTerm

template<i32 Dim>
inline UniformLaplacianRelativeRegTerm<Dim>::UniformLaplacianRelativeRegTerm(const std::vector<i32>& indices, scalar weight, const MatrixNX& refPoints)
    : Super(indices, {}, weight, &refPoints)
{
    i32 nPoints = static_cast<i32>(indices.size());
    this->m_coefficients.reserve(nPoints);
    this->m_coefficients.push_back(scalar(1.0));
    this->m_coefficients.insert(this->m_coefficients.end(), nPoints - 1, scalar(-1.0 / (nPoints - 1)));
}

//// End UniformLaplacianRelativeRegTerm

//// Begin LaplacianRelativeRegTerm

template<i32 Dim>
inline LaplacianRelativeRegTerm<Dim>::LaplacianRelativeRegTerm(const std::vector<i32>& indices, const std::vector<scalar>& coefs, scalar weight, const MatrixNX& refPoints)
    : Super(indices, coefs, weight, &refPoints)
{
}

//// End LaplacianRelativeRegTerm

END_NAMESPACE()
