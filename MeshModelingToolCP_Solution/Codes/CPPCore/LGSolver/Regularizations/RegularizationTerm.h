#pragma once

#include "TypesCommon.h"
#include <vector>
#include <memory>

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
class RegularizationTermAbstract
{
public:
    USING_MATRIX_VECTOR_SHORTNAME(Dim)
public:

    virtual void evaluate(VectorXi& outPointIndices, VectorX& outCoefficients, VectorN& outValue) const = 0;

    virtual ~RegularizationTermAbstract() {}
};

END_NAMESPACE()
