#pragma once

#include "GeometrySolver.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim, typename TOptimizer, typename TSPDLinearSolver>
inline GeometrySolver<Dim, TOptimizer, TSPDLinearSolver>::GeometrySolver()
{
}

template<i32 Dim, typename TOptimizer, typename TSPDLinearSolver>
inline GeometrySolver<Dim, TOptimizer, TSPDLinearSolver>::~GeometrySolver()
{
}

END_NAMESPACE()
