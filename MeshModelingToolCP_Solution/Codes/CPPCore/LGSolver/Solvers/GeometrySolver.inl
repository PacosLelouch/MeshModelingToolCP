#pragma once

#include "GeometrySolver.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
inline scalar GeometrySolverErrorEvaluator<Dim>::evaluate(const MatrixNX& fullQ, ConstraintSetAbstract<Dim>& constraintSet, MatrixNX* outProjectionPtr)
{
    assert(outProjectionPtr != nullptr);

    const std::vector<std::shared_ptr<ConstraintAbstract<Dim> > >& constraints = constraintSet.getConstraints();
    i32 nConstraints = i32(constraints.size());
    i32 nRegTerms = this->m_regularizationMatrixStorage.getNRows();
    scalar error = scalar(0);

    OMP_PARALLEL
    {
        OMP_FOR
        for (i32 i = 0; i < nConstraints; ++i)
        {
            this->m_consError(i) = constraints[i]->project(fullQ, *outProjectionPtr);
        }

        OMP_FOR
        for (i32 i = 0; i < nRegTerms; ++i)
        {
            this->m_regError(i) = this->evaluateRegularizationTerm(i, fullQ);
        }
    }

    error = this->m_consError.sum() + this->m_regError.sum();
    return error;
}

template<i32 Dim, typename TTimer, typename TOptimizer, typename TSPDLinearSolver, typename TRegularizer, typename TConstraintSet>
inline GeometrySolver<Dim, TTimer, TOptimizer, TSPDLinearSolver, TRegularizer, TConstraintSet>::GeometrySolver()
{
}

template<i32 Dim, typename TTimer, typename TOptimizer, typename TSPDLinearSolver, typename TRegularizer, typename TConstraintSet>
inline GeometrySolver<Dim, TTimer, TOptimizer, TSPDLinearSolver, TRegularizer, TConstraintSet>::~GeometrySolver()
{
    static_assert(std::is_base_of_v<LocalGlobalOptimizer<Dim>, TOptimizer>);
}

template<i32 Dim, typename TTimer, typename TOptimizer, typename TSPDLinearSolver, typename TRegularizer, typename TConstraintSet>
inline bool GeometrySolver<Dim, TTimer, TOptimizer, TSPDLinearSolver, TRegularizer, TConstraintSet>::initialize(i32 nPoints, const std::vector<i32>& handleIndices)
{
    this->m_timer.reset();
    TimerUtil::EventID tBegin = this->m_timer.recordTime("GeometrySolver_Initialize_Begin");

    bool result = this->m_optimizer.initialize(nPoints, this->m_constraintSet, this->m_regularizer, &this->m_errorEvaluator, &this->m_linearSolver, handleIndices);

    i32 nRegError = this->m_errorEvaluator.m_regularizationMatrixStorage.getNRows();
    this->m_errorEvaluator.m_regError.setZero(nRegError);

    const std::vector<std::shared_ptr<ConstraintAbstract<Dim> > >& constraints = this->m_constraintSet.getConstraints();
    this->m_errorEvaluator.m_consError.setZero(constraints.size());

    TimerUtil::EventID tEnd = this->m_timer.recordTime("GeometrySolver_Initialize_End");

    // For debugging.
    std::cout << "Initialization time = " << this->m_timer.getElapsedTime(tBegin, tEnd) << std::endl;

    return true;
}

template<i32 Dim, typename TTimer, typename TOptimizer, typename TSPDLinearSolver, typename TRegularizer, typename TConstraintSet>
inline bool GeometrySolver<Dim, TTimer, TOptimizer, TSPDLinearSolver, TRegularizer, TConstraintSet>::solve(i32 nIter, const MatrixNX* initPointsPtr)
{
    if (!isInitialized())
    {
        // Not initialized yet.
        return false;
    }
    if (!initPointsPtr)
    {
        return false;
    }

    this->m_elapsedTimes.clear();
    this->m_funcValues.clear();

    //i32 mAnderson = 5; // TODO, refactor.

    //const MatrixNX& initPoints = *initPointsPtr;

    //bool hasHandles = static_cast<i32>(this->m_handleIndices.size()) > 0;
    //MatrixNX rhsFixedPart = this->m_rhsFixed;
    //if (hasHandles)
    //{

    //    rhsFixedPart += (this->m_rhsHandleContribution * (this->m_handleSelection * initPoints.transpose())).transpose();
    //    this->m_pointsVar1 = (this->m_varSelection * initPoints.transpose()).transpose();
    //    this->m_pointsVar2 = this->m_pointsVar1;
    //    this->m_pointsVar3 = this->m_pointsVar1;
    //}
    //else
    //{
    //    this->m_pointsVar1 = initPoints;
    //    this->m_pointsVar2 = initPoints;
    //    this->m_pointsVar3 = initPoints;
    //}

    //bool accelerate = mAnderson > 0;

    //this->m_curPointsVarPtr = &this->m_pointsVar1;
    //this->m_altPointsVarPtr = &this->m_pointsVar2;
    //this->m_prevPointsVarPtr = &this->m_pointsVar3;
    //this->m_projectionsPtr = &this->m_projections1;
    //if (hasHandles)
    //{
    //    this->m_fullCoords1 = initPoints;
    //    this->m_fullCoordsPtr = &this->m_fullCoords1;
    //}
    //else
    //{
    //    this->m_fullCoordsPtr = this->m_curPointsVarPtr;
    //}

    //i32 nColumns = Dim;
    //i32 nPointsVar = i32(this->m_varPointIndices.size());

    //scalar newError = 0;
    //scalar curError = std::numeric_limits<scalar>::max();

    if (!this->m_optimizer.preBeginOptimization(nIter, initPointsPtr))
    {
        // Fail to initialize optimizer.
        return false;
    }

    //const std::vector<std::shared_ptr<ConstraintAbstract<Dim> > >& constraints = this->m_constraintSet.getConstraints();
    TimerUtil::EventID tBegin = this->m_timer.recordTime("GeometrySolver_Solve_Begin");

    for (i32 it = 0; it <= nIter; ++it)
    {
        if (!this->m_optimizer.optimize(this->m_errorEvaluator, &this->m_constraintSet, &this->m_linearSolver))
        {
            TimerUtil::EventID tError = this->m_timer.recordTime("GeometrySolver_Solve_Error");
            // For debugging.
            std::cout << "Iteration [" << it << "] error time = " << this->m_timer.getElapsedTime(tBegin, tError) << std::endl;
            return false;
        }

        TimerUtil::EventID tIter = this->m_timer.recordTime(("GeometrySolver_Solve_Iter" + std::to_string(it)).c_str());
        this->m_elapsedTimes.push_back(this->m_timer.getElapsedTime(tBegin, tIter));
        this->m_funcValues.push_back(this->m_optimizer.getCurrentError());
        // For debugging.
        std::cout << "Iteration [" << it << "] time = " << this->m_timer.getElapsedTime(tBegin, tIter) << ", error = " << this->m_funcValues.back() << std::endl;
    }

    TimerUtil::EventID tEnd = this->m_timer.recordTime("GeometrySolver_Solve_End");
    // For debugging.
    std::cout << "Iteration end time = " << this->m_timer.getElapsedTime(tBegin, tEnd) << ", error = " << this->m_funcValues.back() << std::endl;

    return true;
}

template<i32 Dim, typename TTimer, typename TOptimizer, typename TSPDLinearSolver, typename TRegularizer, typename TConstraintSet>
inline void GeometrySolver<Dim, TTimer, TOptimizer, TSPDLinearSolver, TRegularizer, TConstraintSet>::getOutput(MatrixNX& output) const
{
    this->m_optimizer.getOutput(output);
}

END_NAMESPACE()
