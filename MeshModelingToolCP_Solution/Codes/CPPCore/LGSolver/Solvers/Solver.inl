#pragma once

#include "Solver.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim, typename TRegularizer, typename TConstraintSet>
inline void SolverAbstract<Dim, TRegularizer, TConstraintSet>::clearConstraints()
{
    m_constraintSet.clearConstraints();
}

template<i32 Dim, typename TRegularizer, typename TConstraintSet>
inline i32 SolverAbstract<Dim, TRegularizer, TConstraintSet>::addConstraint(const std::shared_ptr<ConstraintAbstract<Dim> >& constraintShPtr)
{
    return m_constraintSet.addConstraint(constraintShPtr);
}

template<i32 Dim, typename TRegularizer, typename TConstraintSet>
inline i32 SolverAbstract<Dim, TRegularizer, TConstraintSet>::addRegularizationTerm(const std::shared_ptr<RegularizationTermAbstract<Dim> >& regularizationTermShPtr)
{
    return m_regularizer.addRegularizationTerm(regularizationTermShPtr);
}

template<i32 Dim, typename TOptimizer, typename TSPDLinearSolver, typename TRegularizer, typename TConstraintSet>
SolverBase<Dim, TOptimizer, TSPDLinearSolver, TRegularizer, TConstraintSet>::SolverBase()
{
    static_assert(std::is_base_of_v<OptimizerAbstract<Dim>, TOptimizer>);
    static_assert(!std::is_same_v<OptimizerAbstract<Dim>, TOptimizer>);

    static_assert(std::is_base_of_v<SPDLinearSolverAbstract<Dim>, TSPDLinearSolver>);
    static_assert(!std::is_same_v<SPDLinearSolverAbstract<Dim>, TSPDLinearSolver>);

    static_assert(std::is_base_of_v<RegularizerAbstract<Dim>, TRegularizer>);
    static_assert(!std::is_same_v<RegularizerAbstract<Dim>, TRegularizer>);

    static_assert(std::is_base_of_v<ConstraintSetAbstract<Dim>, TConstraintSet>);
    static_assert(!std::is_same_v<ConstraintSetAbstract<Dim>, TConstraintSet>);
}

END_NAMESPACE()
