#pragma once

#include "TypesCommon.h"
#include <vector>
#include <memory>
#include "RegularizationTerm.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
class RegularizerAbstract
{
public:

    using VectorN = MatrixT<Dim, 1>;
    using MatrixNX = MatrixT<Dim, Eigen::Dynamic>;
    using MatrixXN = MatrixT<Eigen::Dynamic, Dim>;

    i32 addRegularizationTerm(const std::shared_ptr<RegularizationTermAbstract<Dim> >& regularizationTermShPtr);

    const std::vector<std::shared_ptr<RegularizationTermAbstract<Dim> > >& getRegularizationTerms() const;

    void clearRegularizationTerms();

    // Generate data in Regularizer class from regularization terms.
    virtual void generateRegularizationData() = 0;

    // Set up the linear system AX = b for the regularization.
    virtual bool extractRegularizationSystem(i32 nPoints, ColMSMatrix& L, MatrixXN& rightHandSide) const = 0;

protected:
    std::vector<std::shared_ptr<RegularizationTermAbstract<Dim> > > m_regularizationTermShPtrs;
};

template<i32 Dim>
class LinearRegularizer : public RegularizerAbstract<Dim>
{
public:

    using Super = RegularizerAbstract<Dim>;
    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    virtual void generateRegularizationData() override;

    virtual bool extractRegularizationSystem(i32 nPoints, ColMSMatrix& L, MatrixXN& rightHandSide) const override;

    void clearRegularizationData();

protected:
    std::vector<VectorXi> m_pointIndices; // Indices of points s.t. regularization constraints
    std::vector<VectorX> m_coefficients; // Coefficients of the linear combination.
    std::vector<scalar> m_targetValues; // Tarvet values of the linear combination, flatten into a single array. (Using Eigen vector may cause alignment issue.)
};

END_NAMESPACE()

#include "Regularizer.inl"
