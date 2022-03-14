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

    virtual bool initialize(const MatrixNX& data) = 0;

    virtual void assignInputData(const MatrixNX& data) = 0;

    virtual bool optimize(VectorX& uOptimized) = 0;

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

    virtual bool initialize(const MatrixNX& data) override;

    virtual void assignInputData(const MatrixNX& data) override;

    virtual bool optimize(VectorX& uOptimized) override;

    void assignG(const MatrixNX& g);

protected:

    VectorX m_cur_U;

    VectorX m_cur_G;

    i32 m_nDimVar;
    i32 m_accumulateIter;
    i32 m_colIdxHistory;
};

template<i32 Dim>
class AndersonAccelerationOptimizer : public LocalGlobalOptimizer<Dim>
{
public:

    using Super = LocalGlobalOptimizer<Dim>;
    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    virtual bool initialize(const MatrixNX& data) override;

    virtual bool optimize(VectorX& uOptimized) override;

    void setNumberOfHistoryUsed(i32 m) { m_nHistory = m; }
    i32 getNumberOfHistoryUsed() const { return m_nHistory; }

protected:
    i32 m_nHistory = 5;

    VectorX m_cur_F;
    MatrixXX m_prev_dG;
    MatrixXX m_prev_dF;
    MatrixXX m_M;
    VectorX m_theta;
    VectorX m_dF_scale;

    Eigen::CompleteOrthogonalDecomposition<MatrixXX> m_completeOrthoDecomp;
};

END_NAMESPACE()

#include "Optimizer.inl"
