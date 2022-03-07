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

    virtual bool initialize() = 0;

    virtual void solve() = 0;

protected:
};

template<i32 Dim>
class Simplicial_LLT_LinearSolver : public SPDLinearSolverAbstract<Dim>
{
public:
    virtual bool initialize() override;

    virtual void solve() override;
};

END_NAMESPACE()

#include "SPDLinearSolver.inl"
