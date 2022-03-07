#pragma once

#include "Solver.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim,
    typename TOptimizer = LocalGlobalOptimizer<Dim>,
    typename TSPDLinearSolver = Simplicial_LLT_LinearSolver<Dim>,
    typename TRegularizer = LinearRegularizer<Dim>,
    typename TConstraintSet = ConstraintSet<Dim> >
class GeometrySolver : public SolverBase<Dim, 
    TOptimizer, 
    TSPDLinearSolver,
    TRegularizer,
    TConstraintSet>
{
public:

    using Super = SolverBase<Dim,
        TOptimizer,
        TSPDLinearSolver,
        TRegularizer,
        TConstraintSet>;
    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    GeometrySolver();

    virtual ~GeometrySolver();

    virtual bool initialize(i32 nPoints, const std::vector<i32>& fixIndices = std::vector<i32>()) override { return false; }//TODO

    //void solve();

protected:
};

END_NAMESPACE()

#include "GeometrySolver.inl"
