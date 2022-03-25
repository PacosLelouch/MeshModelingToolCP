#pragma once

#include "TypesCommon.h"
#include "LGSolver/Constraints/Constraint.h"
#include "LGSolver/Regularizations/Regularizer.h"
#include "LGSolver/Optimizers/Optimizer.h"
#include "LGSolver/LinearSolvers/SPDLinearSolver.h"
#include <vector>
#include <memory>

BEGIN_NAMESPACE(AAShapeUp)

// The abstract class of solver.
template<i32 Dim, typename TRegularizer = RegularizerAbstract<Dim>, typename TConstraintSet = ConstraintSetAbstract<Dim> >
class SolverAbstract
{
public:
    USING_MATRIX_VECTOR_SHORTNAME(Dim)

    virtual ~SolverAbstract() {}

    void clearConstraints();
    void clearRegularizations();

    TConstraintSet& getConstraintSet() { return m_constraintSet; }
    const TConstraintSet& getConstraintSet() const { return m_constraintSet; }

    TRegularizer& getRegularizer() { return m_regularizer; }
    const TRegularizer& getRegularizer() const { return m_regularizer; }

    i32 addConstraint(const std::shared_ptr<ConstraintAbstract<Dim> >& constraintShPtr);

    i32 addRegularizationTerm(const std::shared_ptr<RegularizationTermAbstract<Dim> >& regularizationTermShPtr);

    virtual bool initialize(i32 nPoints, const std::vector<i32>& handleIndices = std::vector<i32>()) = 0;

    virtual bool solve(i32 nIter, const MatrixNX* initPointsPtr = nullptr) = 0;

    virtual void getOutput(MatrixNX& output) const = 0;

protected:
    TConstraintSet m_constraintSet;
    TRegularizer m_regularizer;

};

// The base class of solver, with some implementations.
template<i32 Dim, 
    typename TTimer = NullTimer,
    typename TOptimizer = OptimizerAbstract<Dim>, // e.g. LocalGlobalOptimizer, AndersonAccelerationOptimizer
    typename TSPDLinearSolver = SPDLinearSolverAbstract<Dim>,  // e.g. Simplicial_LLT_LinearSolver, Simplicial_LDLT_LinearSolver, ConjugateGradientLinearSolver
    typename TRegularizer = RegularizerAbstract<Dim>, // e.g. LinearRegularizer
    typename TConstraintSet = ConstraintSetAbstract<Dim> > // e.g. ConstraintSet in common case
    class SolverBase : public SolverAbstract<Dim, TRegularizer, TConstraintSet>
{
public:
    using Super = SolverAbstract<Dim, TRegularizer, TConstraintSet>;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)

    SolverBase();

    virtual ~SolverBase() {}

    TOptimizer& getOptimizer() { return m_optimizer; }
    const TOptimizer& getOptimizer() const { return m_optimizer; }

    TSPDLinearSolver& getLinearSolver() { return m_linearSolver; }
    const TSPDLinearSolver& getLinearSolver() const { return m_linearSolver; }

    TTimer& getTimer() { return m_timer; }
    const TTimer& getTimer() const { return m_timer; }

protected:
    TOptimizer m_optimizer;
    TSPDLinearSolver m_linearSolver;
    TTimer m_timer;
};

END_NAMESPACE()

#include "Solver.inl"
