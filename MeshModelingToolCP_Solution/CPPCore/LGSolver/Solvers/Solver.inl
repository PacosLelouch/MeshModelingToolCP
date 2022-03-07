#pragma once

#include "Solver.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
inline void SolverAbstract<Dim>::clearConstraints()
{
    m_constraintShPtrs.clear();
}

template<i32 Dim>
inline i32 SolverAbstract<Dim>::addConstraint(const std::shared_ptr<ConstraintAbstract<Dim> >& constraintShPtr)
{
    m_constraintShPtrs.push_back(constraintShPtr);
    return static_cast<i32>(m_constraintShPtrs.size());
}

END_NAMESPACE()
