#pragma once

#include "RegularizationTerm.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
class LaplacianRegTermBase : public RegularizationTermAbstract<Dim>
{
public:
    using Super = RegularizationTermAbstract<Dim>;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)
public:

    LaplacianRegTermBase(const std::vector<i32>& indices, const std::vector<scalar>& coefs, scalar weight, const MatrixNX* refPointsPtr = nullptr);

    virtual ~LaplacianRegTermBase() {}

    virtual void evaluate(VectorXi& outPointIndices, VectorX& outCoefficients, VectorN& outValue) const override;

protected:
    void generateLaplacianHelper(
        VectorXi& outPointIndices, VectorX& outCoefficients, VectorN& outValue,
        const std::vector<int>& indices, const std::vector<scalar>& coefs, scalar weight, const MatrixNX* refPoints = nullptr) const;

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
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)
public:

    UniformLaplacianRegTerm(const std::vector<i32>& indices, scalar weight);

    virtual ~UniformLaplacianRegTerm() {}

protected:
};

template<i32 Dim>
class LaplacianRegTerm : public LaplacianRegTermBase<Dim>
{
public:
    using Super = LaplacianRegTermBase<Dim>;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)
public:

    LaplacianRegTerm(const std::vector<i32>& indices, const std::vector<scalar>& coefs, scalar weight);

    virtual ~LaplacianRegTerm() {}

protected:
};

template<i32 Dim>
class UniformLaplacianRelativeRegTerm : public LaplacianRegTermBase<Dim>
{
public:
    using Super = LaplacianRegTermBase<Dim>;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)
public:

    UniformLaplacianRelativeRegTerm(const std::vector<i32>& indices, scalar weight, const MatrixNX& refPoints);

    virtual ~UniformLaplacianRelativeRegTerm() {}
};


template<i32 Dim>
class LaplacianRelativeRegTerm : public LaplacianRegTermBase<Dim>
{
public:
    using Super = LaplacianRegTermBase<Dim>;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)
public:

    LaplacianRelativeRegTerm(const std::vector<i32>& indices, const std::vector<scalar>& coefs, scalar weight, const MatrixNX& refPoints);

    virtual ~LaplacianRelativeRegTerm() {}
};

END_NAMESPACE()

#include "LaplacianRegTerm.inl"
