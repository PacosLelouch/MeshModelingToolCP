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

    using VectorN = MatrixT<Dim, 1>;
    using MatrixNX = MatrixT<Dim, Eigen::Dynamic>;
    using MatrixXN = MatrixT<Eigen::Dynamic, Dim>;

    virtual ~SolverAbstract() {}

    void clearConstraints();

    i32 addConstraint(const std::shared_ptr<ConstraintAbstract<Dim> >& constraintShPtr);

    i32 addRegularizationTerm(const std::shared_ptr<RegularizationTermAbstract<Dim> >& regularizationTermShPtr);

    virtual bool initialize(i32 nPoints, const std::vector<i32>& fixIndices = std::vector<i32>()) = 0;


protected:
    TConstraintSet m_constraintSet;
    TRegularizer m_regularizer;

};

// The base class of solver, with some implementations.
template<i32 Dim,
    typename TOptimizer = OptimizerAbstract<Dim>, // e.g. LocalGlobalOptimizer, AndersonAccelerationOptimizer
    typename TSPDLinearSolver = SPDLinearSolverAbstract<Dim>,  // e.g. Simplicial_LLT_LinearSolver, Simplicial_LDLT_LinearSolver, ConjugateGradientLinearSolver
    typename TRegularizer = RegularizerAbstract<Dim>, // e.g. LinearRegularizer
    typename TConstraintSet = ConstraintSetAbstract<Dim> > // e.g. ConstraintSet in common case
    class SolverBase : public SolverAbstract<Dim, TRegularizer, TConstraintSet>
{
public:

    using Super = SolverAbstract<Dim, TRegularizer, TConstraintSet>;
    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    SolverBase();

    virtual ~SolverBase() {}

protected:
    TOptimizer m_optimizer;
    TSPDLinearSolver m_linearSolver;
};

END_NAMESPACE()

#include "Solver.inl"
