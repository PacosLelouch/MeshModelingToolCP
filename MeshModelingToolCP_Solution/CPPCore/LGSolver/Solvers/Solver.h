#pragma once

#include "TypesCommon.h"
#include "LGSolver/Constraints/Constraint.h"
#include <vector>
#include <memory>

BEGIN_NAMESPACE(AAShapeUp)

// The abstract class of solver.
template<i32 Dim>
class SolverAbstract
{
public:

    using VectorN = MatrixT<Dim, 1>;
    using MatrixNX = MatrixT<Dim, Eigen::Dynamic>;
    using MatrixXN = MatrixT<Eigen::Dynamic, Dim>;

    virtual ~SolverAbstract() {}

    void clearConstraints();

    i32 addConstraint(const std::shared_ptr<ConstraintAbstract<Dim> >& constraintShPtr);

    virtual bool initialize(i32 nPoints, const std::vector<i32>& fixIndices = std::vector<i32>()) = 0;


protected:
    std::vector<std::shared_ptr<ConstraintAbstract<Dim> > > m_constraintShPtrs;

};

// The base class of solver, with some implementations.
template<i32 Dim,
    typename TOptimizer, // e.g. LocalGlobalOptimizer, AndersonAccelerationOptimizer
    typename TSPDLinearSolver> // e.g. Simplicial_LLT_LinearSolver, Simplicial_LDLT_LinearSolver, ConjugateGradientLinearSolver
    class SolverBase : public SolverAbstract<Dim>
{
public:
    SolverBase() {} // TODO

    virtual ~SolverBase() {}

protected:
};

END_NAMESPACE()

#include "Solver.inl"
