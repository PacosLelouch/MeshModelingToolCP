#pragma once

#include "Optimizer.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
inline bool LocalGlobalOptimizer<Dim>::initialize()
{
    return false;
}

template<i32 Dim>
inline void LocalGlobalOptimizer<Dim>::solve()
{
}

END_NAMESPACE()
