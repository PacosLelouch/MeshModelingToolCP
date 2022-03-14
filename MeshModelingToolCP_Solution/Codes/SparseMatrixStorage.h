#pragma once

#include "TypesCommonMinimal.h"
#include <cassert>

BEGIN_NAMESPACE(AAShapeUp)


struct RowSparseStorageData
{
    void clear()
    {
        m_elements.clear();
        m_rowAddrs.clear();
        m_nRows = 0;
    }

    std::vector<std::pair<i32, scalar> > m_elements;
    std::vector<i32> m_rowAddrs;
    i32 m_nRows = 0;
};

// Contiguous row-wise storage of the element in a sparse matrix
class RowSparseStorage
{
public:

    void clear()
    {
        m_data.clear();
    }

    bool initFromTransposedMatrix(const ColMSMatrix& matTransposed)
    {
        m_data.m_elements.clear();
        m_data.m_rowAddrs.clear();
        m_data.m_nRows = i32(matTransposed.cols());

        for (i64 k = 0; k < matTransposed.outerSize(); ++k)
        {
            m_data.m_rowAddrs.push_back(i32(m_data.m_elements.size()));
            for (ColMSMatrix::InnerIterator it(matTransposed, k); it; ++it)
            {
                m_data.m_elements.push_back(std::make_pair(i32(it.row()), scalar(it.value())));
            }
        }

        m_data.m_rowAddrs.push_back(i32(m_data.m_elements.size()));
        return true;
    }

    template<typename TMatrix, typename TVector>
    void evaluate(i32 rowIdx, const TMatrix& mat, TVector& resVec) const
    {
        i32 addrBegin = m_data.m_rowAddrs[rowIdx], addrEnd = m_data.m_rowAddrs[rowIdx + 1];
        for (i32 i = addrBegin; i < addrEnd; ++i)
        {
            const std::pair<i32, scalar>& curEntry = m_data.m_elements[i];
            resVec += mat.col(curEntry.first) * curEntry.second;
        }
    }

    i32 getNRows() const { return m_data.m_nRows; }

protected:
    RowSparseStorageData m_data;
};

// Row-wise storage of the diagonally dominant matrix for Jacobi iteration
class JacobiRowSparseStorage
{
public:

    void clear()
    {
        m_data.clear();
    }

    bool initFromTransposedMatrix(const ColMSMatrix& matTransposed)
    {
        m_data.m_elements.clear();
        m_data.m_rowAddrs.clear();
        m_data.m_nRows = i32(matTransposed.cols());

        for (i64 k = 0; k < matTransposed.outerSize(); ++k)
        {
            m_data.m_rowAddrs.push_back(i32(m_data.m_elements.size()));
            scalar diagValue = scalar(1);
            scalar offDiagValues = scalar(0);
            bool hasDiag = false;

            for (ColMSMatrix::InnerIterator it(matTransposed, k); it; ++it)
            {
                if (it.row() == it.col())
                {
                    diagValue = it.value();
                    hasDiag = true;
                }
                else
                {
                    m_data.m_elements.push_back(std::make_pair(i32(it.row()), scalar(it.value())));
                    offDiagValues += glm::abs(it.value());
                }
            }

            if (!hasDiag)
            {
                // Error diagonal elements for Jacobi system matrix.
                return false;
            }

            m_data.m_elements.push_back(std::make_pair(-1, scalar(diagValue)));
        }

        m_data.m_rowAddrs.push_back(i32(m_data.m_elements.size()));
        return true;
    }

    template<typename TMatrix, typename TVector>
    void evaluate(i32 rowIdx, const TMatrix& mat, TVector& resVec) const
    {
        i32 addrBegin = m_data.m_rowAddrs[rowIdx], addrEnd = m_data.m_rowAddrs[rowIdx + 1] - 1;
        for (i32 i = addrBegin; i < addrEnd; ++i)
        {
            const std::pair<i32, scalar>& curEntry = m_data.m_elements[i];
            resVec -= mat.col(curEntry.first) * curEntry.second;
        }
        resVec /= m_data.m_elements[addrEnd].second;
    }

    i32 getNRows() const { return m_data.m_nRows; }

    scalar getDiagonal(i32 rowIdx) const { return m_data.m_elements[m_data.m_rowAddrs[rowIdx + 1] - 1].second; }

    void setDiagonal(i32 rowIdx, scalar value) { m_data.m_elements[m_data.m_rowAddrs[rowIdx + 1] - 1].second = value; }

    void getDiagonals(VectorX& diag) const
    {
        diag.resize(m_data.m_nRows);
        for (i32 i = 0; i < m_data.m_nRows; ++i)
        {
            diag(i) = getDiagonal(i);
        }
    }

    void setDiagonals(const VectorX& diag)
    {
        assert(static_cast<i32>(diag.size()) == m_data.m_nRows);
        for (i32 i = 0; i < m_data.m_nRows; ++i)
        {
            setDiagonal(i, diag(i));
        }
    }

protected:
    RowSparseStorageData m_data;
};

END_NAMESPACE()
