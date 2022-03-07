#pragma once

#include "Solver.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim,
    typename TOptimizer,
    typename TSPDLinearSolver>
class GeometrySolver : public SolverBase<Dim, 
    TOptimizer, 
    TSPDLinearSolver>
{
public:
    GeometrySolver();

    virtual ~GeometrySolver();

    virtual bool initialize(i32 nPoints, const std::vector<i32>& fixIndices = std::vector<i32>()) override { return false; }//TODO

    //void solve();

protected:
};

END_NAMESPACE()

#include "GeometrySolver.inl"
