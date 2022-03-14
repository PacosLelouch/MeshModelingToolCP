#pragma once

#include "TypesCommon.h"
#include <vector>

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
class SPDLinearSolverAbstract
{
public:

    using VectorN = MatrixT<Dim, 1>;
    using MatrixNX = MatrixT<Dim, Eigen::Dynamic>;
    using MatrixXN = MatrixT<Eigen::Dynamic, Dim>;

    virtual ~SPDLinearSolverAbstract() {}

    virtual bool initialize(const ColMSMatrix& A) = 0;

    virtual bool solve(const VectorX& b, const VectorX& x0, VectorX& x) const = 0;

    virtual Eigen::ComputationInfo getComputationInfo() const = 0;

protected:
};

template<i32 Dim>
class Simplicial_LLT_LinearSolver : public SPDLinearSolverAbstract<Dim>
{
public:
    using Super = SPDLinearSolverAbstract<Dim>;
    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    virtual ~Simplicial_LLT_LinearSolver();

    virtual bool initialize(const ColMSMatrix& A) override final;

    virtual bool solve(const VectorX& b, const VectorX& x0, VectorX& x) const override final;

    virtual Eigen::ComputationInfo getComputationInfo() const override final;

protected:
    Eigen::SimplicialLLT<ColMSMatrix, Eigen::Lower> m_EigenSolver;
};

template<i32 Dim>
class Simplicial_LDLT_LinearSolver : public SPDLinearSolverAbstract<Dim>
{
public:
    using Super = SPDLinearSolverAbstract<Dim>;
    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    virtual ~Simplicial_LDLT_LinearSolver();

    virtual bool initialize(const ColMSMatrix& A) override final;

    virtual bool solve(const VectorX& b, const VectorX& x0, VectorX& x) const override final;

    virtual Eigen::ComputationInfo getComputationInfo() const override final;

protected:
    Eigen::SimplicialLDLT<ColMSMatrix, Eigen::Lower> m_EigenSolver;
};

template<i32 Dim>
class ConjugateGradientLinearSolver : public SPDLinearSolverAbstract<Dim>
{
public:
    using Super = SPDLinearSolverAbstract<Dim>;
    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    virtual ~ConjugateGradientLinearSolver();

    virtual bool initialize(const ColMSMatrix& A) override final;

    virtual bool solve(const VectorX& b, const VectorX& x0, VectorX& x) const override final;

    virtual Eigen::ComputationInfo getComputationInfo() const override final;

protected:
    Eigen::ConjugateGradient<ColMSMatrix, Eigen::Lower | Eigen::Upper, Eigen::IncompleteLUT<scalar> > m_EigenSolver;
};

END_NAMESPACE()

#include "SPDLinearSolver.inl"
