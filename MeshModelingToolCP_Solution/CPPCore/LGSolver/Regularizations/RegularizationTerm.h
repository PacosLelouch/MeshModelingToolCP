#pragma once

#include "TypesCommon.h"
#include <vector>
#include <memory>

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
class RegularizationTermAbstract
{
public:
    using VectorN = MatrixT<Dim, 1>;
    using MatrixNX = MatrixT<Dim, Eigen::Dynamic>;
    using MatrixXN = MatrixT<Eigen::Dynamic, Dim>;

    virtual void evaluate(VectorXi& outPointIndices, VectorX& outCoefficients, VectorN& outValue) const = 0;
};

END_NAMESPACE()
