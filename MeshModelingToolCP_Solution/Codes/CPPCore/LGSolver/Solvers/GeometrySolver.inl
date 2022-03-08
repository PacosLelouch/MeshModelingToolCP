#pragma once

#include "GeometrySolver.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim, typename TOptimizer, typename TSPDLinearSolver, typename TRegularizer, typename TConstraintSet>
inline GeometrySolver<Dim, TOptimizer, TSPDLinearSolver, TRegularizer, TConstraintSet>::GeometrySolver()
{
}

template<i32 Dim, typename TOptimizer, typename TSPDLinearSolver, typename TRegularizer, typename TConstraintSet>
inline GeometrySolver<Dim, TOptimizer, TSPDLinearSolver, TRegularizer, TConstraintSet>::~GeometrySolver()
{
}

END_NAMESPACE()
