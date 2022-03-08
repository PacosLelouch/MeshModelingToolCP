#pragma once

#include "TypesCommon.h"
#include <vector>

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
class OptimizerAbstract
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
class LocalGlobalOptimizer : public OptimizerAbstract<Dim>
{
public:

    using Super = OptimizerAbstract<Dim>;
    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    virtual bool initialize() override;

    virtual void solve() override;
};

END_NAMESPACE()

#include "Optimizer.inl"
