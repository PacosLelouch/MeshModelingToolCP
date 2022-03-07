#pragma once

#include "RegularizationTerm.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
class LaplacianRegTermBase : public RegularizationTermAbstract<Dim>
{
public:
    using Super = RegularizationTermAbstract<Dim>;

    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    LaplacianRegTermBase(const std::vector<i32>& indices, const std::vector<scalar>& coefs, scalar weight, const MatrixNX* refPointsPtr = nullptr);

    virtual void evaluate(VectorXi& outPointIndices, VectorX& outCoefficients, VectorN& outValue) const override;

protected:
    void generateLaplacianHelper(
        VectorXi& outPointIndices, VectorX& outCoefficients, VectorN& outValue,
        const std::vector<int>& indices, const std::vector<scalar>& coefs, scalar weight, const MatrixNX* refPoints = nullptr);

protected:
    std::vector<i32> m_pointIndices;
    std::vector<scalar> m_coefficients;
    scalar m_weight;
    const MatrixNX* m_refPointsPtr;
};

template<i32 Dim>
class UniformLaplacianRegTerm : public LaplacianRegTermBase<Dim>
{
public:
    using Super = LaplacianRegTermBase<Dim>;

    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    UniformLaplacianRegTerm(const std::vector<i32>& indices, scalar weight);

protected:
};

template<i32 Dim>
class LaplacianRegTerm : public LaplacianRegTermBase<Dim>
{
public:
    using Super = LaplacianRegTermBase<Dim>;

    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    LaplacianRegTerm(const std::vector<i32>& indices, const std::vector<scalar>& coefs, scalar weight);

protected:
};

template<i32 Dim>
class UniformLaplacianRelativeRegTerm : public LaplacianRegTermBase<Dim>
{
public:
    using Super = LaplacianRegTermBase<Dim>;

    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    UniformLaplacianRelativeRegTerm(const std::vector<i32>& indices, scalar weight, const MatrixNX& refPoints);
};


template<i32 Dim>
class LaplacianRelativeRegTerm : public LaplacianRegTermBase<Dim>
{
public:
    using Super = LaplacianRegTermBase<Dim>;

    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    LaplacianRelativeRegTerm(const std::vector<i32>& indices, scalar weight, const MatrixNX& refPoints);
};

END_NAMESPACE()

#include "LaplacianRegTerm.inl"
