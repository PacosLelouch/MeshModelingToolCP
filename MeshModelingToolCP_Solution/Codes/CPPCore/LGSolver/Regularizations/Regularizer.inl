#pragma once

#include "Regularizer.h"

BEGIN_NAMESPACE(AAShapeUp)

//// Begin RegularizerAbstract

template<i32 Dim>
inline i32 RegularizerAbstract<Dim>::addRegularizationTerm(const std::shared_ptr<RegularizationTermAbstract<Dim> >& regularizationTermShPtr)
{
    m_regularizationTermShPtrs.push_back(regularizationTermShPtr);
    return static_cast<i32>(m_regularizationTermShPtrs.size());
}

template<i32 Dim>
inline const std::vector<std::shared_ptr<RegularizationTermAbstract<Dim>>>& RegularizerAbstract<Dim>::getRegularizationTerms() const
{
    return m_regularizationTermShPtrs;
}

template<i32 Dim>
inline void RegularizerAbstract<Dim>::clearRegularizationTerms()
{
    m_regularizationTermShPtrs.clear();
}

//// End RegularizerAbstract

//// Begin LinearRegularizer

template<i32 Dim>
inline void LinearRegularizer<Dim>::generateRegularizationData()
{
    this->clearRegularizationData();
    for (std::shared_ptr<RegularizationTermAbstract<Dim> >& regTermShPtr : this->m_regularizationTermShPtrs)
    {
        auto& newIndices = this->m_pointIndices.emplace_back();
        auto& newCoef = this->m_coefficients.emplace_back();
        VectorN newTargetValue;

        regTermShPtr->evaluate(newIndices, newCoef, newTargetValue);

        for (i32 i = 0; i < Dim; ++i)
        {
            this->m_targetValues.push_back(newTargetValue(i));
        }
    }
}

template<i32 Dim>
inline bool LinearRegularizer<Dim>::extractRegularizationSystem(i32 nPoints, ColMSMatrix& L, MatrixXN& rightHandSide) const
{
    i32 nRows = static_cast<i32>(m_pointIndices.size());
    rightHandSide.setZero(nRows, Dim);
    L.resize(nRows, nPoints);

    if (nRows == 0)
    {
        return false;
    }

    std::vector<SMatrixTriplet> triplets;
    for (i32 i = 0; i < nRows; ++i)
    {
        i32 nCoefs = static_cast<i32>(m_pointIndices[i].size());
        for (i32 j = 0; j < nCoefs; ++j)
        {
            triplets.push_back(SMatrixTriplet(i, m_pointIndices[i](j), m_coefficients[i](j)));
        }

        Eigen::Map<const VectorX> targetValuesRemapping(&m_targetValues[i * Dim], Dim);
        rightHandSide.row(i) = targetValuesRemapping.transpose();
    }

    L.setFromTriplets(triplets.begin(), triplets.end());
    L.makeCompressed();

    return true;
}

template<i32 Dim>
inline void LinearRegularizer<Dim>::clearRegularizationData()
{
    m_pointIndices.clear();
    m_coefficients.clear();
    m_targetValues.clear();
}

//// End LinearRegularizer

END_NAMESPACE()
