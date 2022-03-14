#pragma once

#include "Optimizer.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
inline bool LocalGlobalOptimizer<Dim>::initialize(const MatrixNX& data)
{
    this->m_nDimVar = i32(data.size());
    this->m_accumulateIter = 0;
    this->m_colIdxHistory = 0;

    this->m_cur_U.resize(this->m_nDimVar);
    this->assignInputData(data);

    return true;
}

template<i32 Dim>
inline void LocalGlobalOptimizer<Dim>::assignInputData(const MatrixNX& data)
{
    this->m_cur_U = Eigen::Map<const VectorX>(data.data(), this->m_nDimVar);
}

template<i32 Dim>
inline bool LocalGlobalOptimizer<Dim>::optimize(VectorX& uOptimized)
{
    //TODO
    return false;
}

template<i32 Dim>
inline void LocalGlobalOptimizer<Dim>::assignG(const MatrixNX& g)
{
    this->m_cur_G = Eigen::Map<const VectorX>(g.data(), this->m_nDimVar);
}

template<i32 Dim>
inline bool AndersonAccelerationOptimizer<Dim>::initialize(const MatrixNX& data)
{
    assert(m_nHistory > 0);
    if (!this->Super::initialize(data))
    {
        return false;
    }
    const i32& d = this->m_nDimVar;
    const i32& m = this->m_nHistory;
    this->m_cur_F.resize(d);
    this->m_prev_dG.resize(d, m);
    this->m_prev_dF.resize(d, m);
    this->m_M.resize(m, m);
    this->m_theta.resize(m);
    this->m_dF_scale.resize(m);

    return true;
}

template<i32 Dim>
inline bool AndersonAccelerationOptimizer<Dim>::optimize(VectorX& uOptimized)
{
    assert(this->m_accumulateIter >= 0);

    this->m_cur_F = this->m_cur_G - this->m_cur_U;

    if (this->m_accumulateIter == 0)
    {
        this->m_prev_dF.col(0) = -this->m_cur_F;
        this->m_prev_dG.col(0) = -this->m_cur_G;
        this->m_cur_U = this->m_cur_G;
    }
    else
    {
        this->m_prev_dF.col(this->m_colIdxHistory) += this->m_cur_F;
        this->m_prev_dG.col(this->m_colIdxHistory) += this->m_cur_G;

        scalar eps = glm::epsilon<scalar>();
        scalar scale = glm::max(eps, this->m_prev_dF.col(this->m_colIdxHistory).norm());
        this->m_dF_scale(this->m_colIdxHistory) = scale;
        this->m_prev_dF.col(this->m_colIdxHistory) /= scale;

        i32 mK = glm::min(this->m_nHistory, this->m_accumulateIter);

        if (mK == 1)
        {
            this->m_theta(0) = 0;
            scalar dF_sqrnorm = this->m_prev_dF.col(this->m_colIdxHistory).squaredNorm();
            this->m_M(0, 0) = dF_sqrnorm;
            scalar dF_norm = glm::sqrt(dF_sqrnorm);

            if (dF_norm > eps)
            {
                this->m_theta(0) = (this->m_prev_dF.col(this->m_colIdxHistory) / dF_norm).dot(this->m_cur_F / dF_norm);
            }
        }
        else
        {
            // Update the normal equation matrix, for the column and row corresponding to the new dF column.
            VectorX newInnerProduction = (this->m_prev_dF.col(this->m_colIdxHistory).transpose() * this->m_prev_dF.block(0, 0, this->m_nDimVar, mK)).transpose();
            this->m_M.block(this->m_colIdxHistory, 0, 1, mK) = newInnerProduction.transpose();
            this->m_M.block(0, this->m_colIdxHistory, mK, 1) = newInnerProduction;

            // Solve normal equation
            this->m_completeOrthoDecomp.compute(this->m_M.block(0, 0, mK, mK));
            this->m_theta.head(mK) = this->m_completeOrthoDecomp.solve(this->m_prev_dF.block(0, 0, this->m_nDimVar, mK).transpose() * this->m_cur_F);
        }

        // Use rescaled theta to compute new U.
        this->m_cur_U = this->m_cur_G - this->m_prev_dG.block(0, 0, this->m_nDimVar, mK) * ((this->m_theta.head(mK).array() / this->m_dF_scale.head(mK).array()).matrix());

        this->m_colIdxHistory = (this->m_colIdxHistory + 1) % this->m_nHistory;
        this->m_prev_dF.col(this->m_colIdxHistory) = -this->m_cur_F;
        this->m_prev_dG.col(this->m_colIdxHistory) = -this->m_cur_G;
    }

    ++this->m_accumulateIter;
    uOptimized = this->m_cur_U;
    return true;
}


END_NAMESPACE()
