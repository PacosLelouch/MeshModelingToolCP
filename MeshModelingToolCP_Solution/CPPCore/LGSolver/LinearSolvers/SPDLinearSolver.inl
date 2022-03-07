#pragma once

#include "SPDLinearSolver.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
inline bool Simplicial_LLT_LinearSolver<Dim>::initialize()
{
    return false;
}

template<i32 Dim>
inline void Simplicial_LLT_LinearSolver<Dim>::solve()
{
}

END_NAMESPACE()
