#pragma once

#include "SPDLinearSolver.h"

BEGIN_NAMESPACE(AAShapeUp)

//// Begin Simplicial_LLT_LinearSolver

template<i32 Dim>
inline Simplicial_LLT_LinearSolver<Dim>::~Simplicial_LLT_LinearSolver()
{
}


template<i32 Dim>
inline bool Simplicial_LLT_LinearSolver<Dim>::initialize(const ColMSMatrix& A)
{
    m_EigenSolver.compute(A);
    return m_EigenSolver.info() == Eigen::Success;
}

template<i32 Dim>
inline bool Simplicial_LLT_LinearSolver<Dim>::solve(const VectorX& b, const VectorX& x0, VectorX& x) const
{
    x = m_EigenSolver.solve(b);
    return m_EigenSolver.info() == Eigen::Success;
}

template<i32 Dim>
inline Eigen::ComputationInfo Simplicial_LLT_LinearSolver<Dim>::getComputationInfo() const
{
    return m_EigenSolver.info();
}

//// End Simplicial_LLT_LinearSolver

//// Begin Simplicial_LDLT_LinearSolver

template<i32 Dim>
inline Simplicial_LDLT_LinearSolver<Dim>::~Simplicial_LDLT_LinearSolver()
{
}

template<i32 Dim>
inline bool Simplicial_LDLT_LinearSolver<Dim>::initialize(const ColMSMatrix& A)
{
    m_EigenSolver.compute(A);
    return m_EigenSolver.info() == Eigen::Success;
}

template<i32 Dim>
inline bool Simplicial_LDLT_LinearSolver<Dim>::solve(const VectorX& b, const VectorX& x0, VectorX& x) const
{
    x = m_EigenSolver.solve(b);
    return m_EigenSolver.info() == Eigen::Success;
}

template<i32 Dim>
inline Eigen::ComputationInfo Simplicial_LDLT_LinearSolver<Dim>::getComputationInfo() const
{
    return m_EigenSolver.info();
}

//// End Simplicial_LDLT_LinearSolver

//// Start ConjugateGradientLinearSolver

template<i32 Dim>
inline ConjugateGradientLinearSolver<Dim>::~ConjugateGradientLinearSolver()
{
}

template<i32 Dim>
inline bool ConjugateGradientLinearSolver<Dim>::initialize(const ColMSMatrix& A)
{
    m_EigenSolver.compute(A);
    return m_EigenSolver.info() == Eigen::Success;
}

template<i32 Dim>
inline bool ConjugateGradientLinearSolver<Dim>::solve(const VectorX& b, const VectorX& x0, VectorX& x) const
{
    x = m_EigenSolver.solveWithGuess(b, x0);
    return m_EigenSolver.info() == Eigen::Success;
}

template<i32 Dim>
inline Eigen::ComputationInfo ConjugateGradientLinearSolver<Dim>::getComputationInfo() const
{
    return m_EigenSolver.info();
}

//// End ConjugateGradientLinearSolver

END_NAMESPACE()
