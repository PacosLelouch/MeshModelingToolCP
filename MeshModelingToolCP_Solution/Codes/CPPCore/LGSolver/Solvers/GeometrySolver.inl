#pragma once

#include "GeometrySolver.h"

BEGIN_NAMESPACE(AAShapeUp)

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

    //assert(std::unordered_set<i32>(handleIndices.begin(), handleIndices.end()).size() == handleIndices.size());
    if (!(std::unordered_set<i32>(handleIndices.begin(), handleIndices.end()).size() == handleIndices.size()))
    {
        // Duplicate indices in handle array.
        return false;
    }

    i32 nHandles = i32(handleIndices.size());
    m_handleIndices.resize(nHandles);
    if (nHandles > 0)
    {
        m_handleIndices = Eigen::Map<const VectorXi>(handleIndices.data(), handleIndices.size());
    }
    const std::vector<std::shared_ptr<ConstraintAbstract<Dim> > >& constraintSet = this->m_constraintSet.getConstraints();
    i32 nConstraints = i32(constraintSet.size());

    assert(nPoints > 0);
    assert(nConstraints > 0);

    i32 nVarPoints = nPoints - nHandles;

    this->m_pointsVar1.setZero(Dim, nVarPoints);
    this->m_pointsVar2.setZero(Dim, nVarPoints);
    this->m_pointsVar3.setZero(Dim, nVarPoints);

    if (nHandles > 0)
    {
        this->m_fullCoords1.setZero(Dim, nPoints);
    }

    // Set up full constraint matrix.
    std::vector<SMatrixTriplet> triplets;
    i32 accumConsIdx = 0;
    for (i32 i = 0; i < nConstraints; ++i)
    {
        constraintSet[i]->extractConstraint(triplets, accumConsIdx);
    }
    i32 nProjection = accumConsIdx;

    // Set up full global update matrix.
    ColMSMatrix A(nProjection, nPoints);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
    ColMSMatrix AT = A.transpose();
    ColMSMatrix globalMat = AT * A;

    // Set up regularization terms.
    this->m_rhsFixed.setZero(Dim, nVarPoints);
    MatrixXN fullRhsFixed(nPoints, Dim);
    fullRhsFixed.setZero();
    ColMSMatrix L;
    if (this->m_regularizer.extractRegularizationSystem(nPoints, L, this->m_regularizationRhs))
    {
        ColMSMatrix LT = L.transpose();
        globalMat += LT * L;
        fullRhsFixed = LT * this->m_regularizationRhs;
        this->m_regularizationMatrixStorage.initFromTransposedMatrix(LT);
    }
    else
    {
        this->m_regularizationMatrixStorage.clear();
    }

    this->m_projections1.setZero(Dim, nProjection);
    this->m_consError.setZero(constraintSet.size());
    this->m_regError.setZero(L.rows());
    this->m_lbfgsError.setZero(nVarPoints);

    // Reduce the full system to a system about variables.
    ColMSMatrix AT_Reduced;
    if (nHandles == 0)
    {
        AT_Reduced = AT;
        this->m_rhsFixed = fullRhsFixed.transpose();
        this->m_varPointIndices.resize(nPoints);
        for (i32 i = 0; i < nPoints; ++i)
        {
            this->m_varPointIndices(i) = i;
        }
    }
    else
    {
        std::vector<bool> isHandles(nPoints, false);
        for (i32 i = 0; i < nHandles; ++i)
        {
            isHandles[this->m_handleIndices[i]] = true;
        }

        // Set up selection matrix for handles and variables.
        std::vector<i32> varPointIdxs;
        std::vector<SMatrixTriplet> handleTriplets, varPointTriplets;
        i32 handleRow = 0, varRow = 0;
        for (i32 i = 0; i < nPoints; ++i)
        {
            if (isHandles[i])
            {
                handleTriplets.push_back(SMatrixTriplet(handleRow++, i, scalar(1)));
            }
            else
            {
                varPointIdxs.push_back(i);
                varPointTriplets.push_back(SMatrixTriplet(varRow++, i, scalar(1)));
            }
        }

        this->m_varPointIndices = Eigen::Map<VectorXi>(varPointIdxs.data(), varPointIdxs.size());
        assert(nVarPoints == varRow);
        assert(nHandles == handleRow);

        this->m_varSelection.resize(nVarPoints, nPoints);
        this->m_varSelection.setFromTriplets(varPointTriplets.begin(), varPointTriplets.end());
        this->m_varSelection.makeCompressed();

        this->m_handleSelection.resize(nHandles, nPoints);
        this->m_handleSelection.setFromTriplets(handleTriplets.begin(), handleTriplets.end());
        this->m_handleSelection.makeCompressed();

        ColMSMatrix varToFull = this->m_varSelection.transpose();
        ColMSMatrix handleToFull = this->m_handleSelection.transpose();

        this->m_rhsHandleContribution = -this->m_handleSelection * globalMat * handleToFull;
        globalMat = this->m_varSelection * (globalMat * varToFull);
        this->m_rhsFixed = (this->m_varSelection * fullRhsFixed).transpose();
        AT_Reduced = this->m_varSelection * AT;
    }

    this->m_AT = AT_Reduced;
    if (!this->m_linearSolver.initialize(globalMat))
    {
        // SPD solver initialization failed.
        return false;
    }

    this->m_handleSolverInitialized = nHandles > 0;
    this->m_solverInitialized = nHandles == 0;

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
    i32 mAnderson = 5; // TODO, refactor.

    const MatrixNX& initPoints = *initPointsPtr;

    bool hasHandles = static_cast<i32>(this->m_handleIndices.size()) > 0;
    MatrixNX rhsFixedPart = this->m_rhsFixed;
    if (hasHandles)
    {

        rhsFixedPart += (this->m_rhsHandleContribution * (this->m_handleSelection * initPoints.transpose())).transpose();
        this->m_pointsVar1 = (this->m_varSelection * initPoints.transpose()).transpose();
        this->m_pointsVar2 = this->m_pointsVar1;
        this->m_pointsVar3 = this->m_pointsVar1;
    }
    else
    {
        this->m_pointsVar1 = initPoints;
        this->m_pointsVar2 = initPoints;
        this->m_pointsVar3 = initPoints;
    }

    bool accelerate = mAnderson > 0;

    this->m_curPointsVarPtr = &this->m_pointsVar1;
    this->m_altPointsVarPtr = &this->m_pointsVar2;
    this->m_prevPointsVarPtr = &this->m_pointsVar3;
    this->m_projectionsPtr = &this->m_projections1;
    if (hasHandles)
    {
        this->m_fullCoords1 = initPoints;
        this->m_fullCoordsPtr = &this->m_fullCoords1;
    }
    else
    {
        this->m_fullCoordsPtr = this->m_curPointsVarPtr;
    }

    i32 nColumns = Dim;
    i32 nPointsVar = i32(this->m_varPointIndices.size());

    scalar newError = 0;
    scalar curError = std::numeric_limits<scalar>::max();

    if (!this->m_optimizer.initialize(*this->m_curPointsVarPtr))
    {
        // Fail to initialize optimizer.
        return false;
    }

    const std::vector<std::shared_ptr<ConstraintAbstract<Dim> > >& constraintSet = this->m_constraintSet.getConstraints();

    TimerUtil::EventID tBegin = this->m_timer.recordTime("GeometrySolver_Solve_Begin");

    for (i32 it = 0; it <= nIter; ++it)
    {
        i32 nConstraints = i32(constraintSet.size());
        i32 nRegTerms = this->m_regularizationMatrixStorage.getNRows();
        i32 nFixedCoords = hasHandles ? nPointsVar : 0;

        i32 nPostConstraints = 0;
        i32 nPostRegTerms = 0;
        i32 nPostFixedCoords = 0;

        bool newErrorNeedsUpdate = false;

        OMP_PARALLEL
        {
            OMP_FOR
            for (i32 i = 0; i < nFixedCoords; ++i)
            {
                this->m_fullCoordsPtr->col(this->m_varPointIndices(i)) = this->m_curPointsVarPtr->col(i);
            }
            
            OMP_FOR
            for (i32 i = 0; i < nConstraints; ++i)
            {
                this->m_consError(i) = constraintSet[i]->project(*this->m_fullCoordsPtr, *this->m_projectionsPtr);
            }

            OMP_FOR
            for (i32 i = 0; i < nRegTerms; ++i)
            {
                this->m_regError(i) = this->evaluateRegularizationTerm(i, *this->m_fullCoordsPtr);
            }

            OMP_SECTIONS
            {
                OMP_SECTION
                {
                    newError = this->m_consError.sum() + this->m_regError.sum();
                }
            }

            OMP_SINGLE
            {
                bool requireSwap = (it > 0) && accelerate && (newError > curError);
                if (requireSwap)
                {
                    std::swap(this->m_curPointsVarPtr, this->m_altPointsVarPtr);
                    this->m_optimizer.assignInputData(*this->m_curPointsVarPtr);

                    if (!hasHandles)
                    {
                        this->m_fullCoordsPtr = this->m_curPointsVarPtr;
                    }

                    nPostConstraints = nConstraints;
                    nPostFixedCoords = nFixedCoords;
                    nPostRegTerms = nRegTerms;
                    newErrorNeedsUpdate = true;
                }
                else
                {
                    curError = newError;
                }

                this->m_AndersonReset.push_back(requireSwap);
            }
            
            // If the new error is not smaller than the previous one, re-projection.
            // Else, skip this step.
            OMP_FOR
            for (i32 i = 0; i < nPostFixedCoords; ++i)
            {
                this->m_fullCoordsPtr->col(this->m_varPointIndices(i)) = this->m_curPointsVarPtr->col(i);
            }

            OMP_FOR
            for (i32 i = 0; i < nPostConstraints; ++i)
            {
                this->m_consError(i) = constraintSet[i]->project(*this->m_fullCoordsPtr, *this->m_projectionsPtr);
            }

            OMP_FOR
            for (i32 i = 0; i < nPostRegTerms; ++i)
            {
                this->m_regError(i) = this->evaluateRegularizationTerm(i, *this->m_fullCoordsPtr);
            }

            OMP_SINGLE
            {
                if (newErrorNeedsUpdate)
                {
                    curError = this->m_consError.sum() + this->m_regError.sum();
                }

                // If reach the last iteration, set nRows = 0 to skip the following update.
                if (it == nIter)
                {
                    nColumns = 0;
                }
                else
                {
                    std::swap(this->m_prevPointsVarPtr, this->m_curPointsVarPtr);
                }

                TimerUtil::EventID tIter = this->m_timer.recordTime(("GeometrySolver_Solve_Iter" + std::to_string(it)).c_str());
                this->m_elapsedTimes.push_back(this->m_timer.getElapsedTime(tBegin, tIter));
                this->m_funcValues.push_back(curError);
            }

            // Solve for new points.
            OMP_FOR
            for (i32 i = 0; i < nColumns; ++i)
            {
                VectorX solution;
                solution.resize(this->m_altPointsVarPtr->cols());
                this->m_linearSolver.solve(
                    rhsFixedPart.row(i).transpose() + this->m_AT * this->m_projectionsPtr->row(i).transpose(), 
                    this->m_prevPointsVarPtr->row(i).transpose(), 
                    solution);
                this->m_altPointsVarPtr->row(i) = solution.transpose();
            }

            // Anderson acceleration.
            if (it != nIter)
            {
                if (accelerate)
                {
                    VectorX solution;
                    solution.resize(this->m_curPointsVarPtr->size());
                    this->m_optimizer.assignG(*this->m_altPointsVarPtr);
                    this->m_optimizer.optimize(solution);
                    (*this->m_curPointsVarPtr) = Eigen::Map<const MatrixNX>(solution.data(), Dim, this->m_curPointsVarPtr->cols());
                }
                else
                {
                    std::swap(this->m_curPointsVarPtr, this->m_altPointsVarPtr);
                }
            }

            if (!hasHandles)
            {
                this->m_fullCoordsPtr = this->m_curPointsVarPtr;
            }
        }
        return true;
    }

    TimerUtil::EventID tEnd = this->m_timer.recordTime("GeometrySolver_Solve_End");

    return true;
}

END_NAMESPACE()
