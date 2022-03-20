#pragma once

#include "Optimizer.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
inline bool LocalGlobalOptimizer<Dim>::initialize(i32 nPoints, ConstraintSetAbstract<Dim>& constraintSet, RegularizerAbstract<Dim>& regularizer,
    ErrorEvaluatorAbstract<Dim>* errorEvaluatorPtr, SPDLinearSolverAbstract<Dim>* linearSolverPtr, const std::vector<i32>& handleIndices)
{
    return false;//TODO
}

template<i32 Dim>
inline bool LocalGlobalOptimizer<Dim>::isInitialized() const
{
    return false;//TODO
}

template<i32 Dim>
inline bool LocalGlobalOptimizer<Dim>::preBeginOptimization(i32 nIter, const MatrixNX* initPointsPtr)
{
    this->m_nIter = nIter;

    const MatrixNX& inQ0 = this->m_pointsVar1; // *this->m_curPointsVarPtr;
    this->m_nDimVar = i32(inQ0.size());
    this->m_accumulateIter = 0;
    this->m_colIdxHistory = 0;

    this->m_cur_Q.resize(this->m_nDimVar);
    this->assignInputData(inQ0);

    return true;
}

template<i32 Dim>
inline void LocalGlobalOptimizer<Dim>::assignInputData(const MatrixNX& inQ)
{
    this->m_cur_Q = Eigen::Map<const VectorX>(inQ.data(), this->m_nDimVar);
}

template<i32 Dim>
inline bool LocalGlobalOptimizer<Dim>::optimize(ErrorEvaluatorAbstract<Dim>& errorEvaluator, ConstraintSetAbstract<Dim>* constraintSetPtr, SPDLinearSolverAbstract<Dim>* linearSolverPtr)
{
    //TODO
    return false;
}

template<i32 Dim>
inline void LocalGlobalOptimizer<Dim>::getOutput(MatrixNX& outQn) const
{
    //TODO
}

//template<i32 Dim>
//inline void LocalGlobalOptimizer<Dim>::assignG(const MatrixNX& inG)
//{
//    this->m_cur_G = Eigen::Map<const VectorX>(inG.data(), this->m_nDimVar);
//}

template<i32 Dim>
inline bool AndersonAccelerationOptimizer<Dim>::initialize(i32 nPoints, ConstraintSetAbstract<Dim>& constraintSet, RegularizerAbstract<Dim>& regularizer,
    ErrorEvaluatorAbstract<Dim>* errorEvaluatorPtr, SPDLinearSolverAbstract<Dim>* linearSolverPtr, const std::vector<i32>& handleIndices)
{
    if (!errorEvaluatorPtr)
    {
        return false;
    }
    if (!linearSolverPtr)
    {
        return false;
    }
    //assert(std::unordered_set<i32>(handleIndices.begin(), handleIndices.end()).size() == handleIndices.size());
    if (!(std::unordered_set<i32>(handleIndices.begin(), handleIndices.end()).size() == handleIndices.size()))
    {
        // Duplicate indices in handle array.
        return false;
    }

    i32 nHandles = i32(handleIndices.size());
    this->m_handleIndices.resize(nHandles);
    if (nHandles > 0)
    {
        this->m_handleIndices = Eigen::Map<const VectorXi>(handleIndices.data(), handleIndices.size());
    }
    const std::vector<std::shared_ptr<ConstraintAbstract<Dim> > >& constraints = constraintSet.getConstraints();
    i32 nConstraints = i32(constraints.size());

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
        constraints[i]->extractConstraint(triplets, accumConsIdx);
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
    if (regularizer.extractRegularizationSystem(nPoints, L, errorEvaluatorPtr->m_regularizationRhs))
    {
        ColMSMatrix LT = L.transpose();
        globalMat += LT * L;
        fullRhsFixed = LT * errorEvaluatorPtr->m_regularizationRhs;
        errorEvaluatorPtr->m_regularizationMatrixStorage.initFromTransposedMatrix(LT);
    }
    else
    {
        errorEvaluatorPtr->m_regularizationMatrixStorage.clear();
    }

    this->m_projections1.setZero(Dim, nProjection);
    //errorEvaluatorPtr->m_consError.setZero(constraints.size()); // Init at solver.
    //errorEvaluatorPtr->m_regError.setZero(L.rows()); // Init at solver.

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
    if (!linearSolverPtr->initialize(globalMat))
    {
        // SPD solver initialization failed.
        return false;
    }

    this->m_handleSolverInitialized = nHandles > 0;
    this->m_solverInitialized = nHandles == 0;

    return true;
}

template<i32 Dim>
inline bool AndersonAccelerationOptimizer<Dim>::isInitialized() const
{
    return this->m_solverInitialized || this->m_handleSolverInitialized;
}

template<i32 Dim>
inline bool AndersonAccelerationOptimizer<Dim>::preBeginOptimization(i32 nIter, const MatrixNX* initPointsPtr)
{
    assert(m_mAnderson > 0);

    const MatrixNX& initPoints = *initPointsPtr;

    bool hasHandles = static_cast<i32>(this->m_handleIndices.size()) > 0;
    this->m_rhsFixedWithInitPoints = this->m_rhsFixed;
    if (hasHandles)
    {

        this->m_rhsFixedWithInitPoints += (this->m_rhsHandleContribution * (this->m_handleSelection * initPoints.transpose())).transpose();
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

    bool accelerate = m_mAnderson > 0;

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

    this->m_newError = scalar(0);
    this->m_curError = std::numeric_limits<scalar>::max();

    if (!this->Super::preBeginOptimization(nIter, initPointsPtr))
    {
        return false;
    }
    const i32& d = this->m_nDimVar;
    const i32& m = this->m_mAnderson;
    this->m_cur_F.resize(d);
    this->m_prev_dG.resize(d, m);
    this->m_prev_dF.resize(d, m);
    this->m_M.resize(m, m);
    this->m_theta.resize(m);
    this->m_dF_scale.resize(m);

    return true;
}

template<i32 Dim>
inline bool AndersonAccelerationOptimizer<Dim>::optimize(ErrorEvaluatorAbstract<Dim>& errorEvaluator, ConstraintSetAbstract<Dim>* constraintSetPtr, SPDLinearSolverAbstract<Dim>* linearSolverPtr)
{
    if (!constraintSetPtr)
    {
        return false;
    }
    if (!linearSolverPtr)
    {
        return false;
    }

    bool hasHandles = static_cast<i32>(this->m_handleIndices.size()) > 0;
    bool accelerate = this->m_mAnderson > 0;

    i32 nColumns = Dim;
    i32 nPointsVar = i32(this->m_varPointIndices.size());

    i32 nFixedCoords = hasHandles ? nPointsVar : 0;

    i32 it = this->m_accumulateIter;

    bool newErrorNeedsUpdate = false;

    //OMP_PARALLEL
    {
        // Compute full coordinates and error. (Error as a function or a class?)
        //OMP_FOR
        OMP_PARALLEL_(for)
        for (i32 i = 0; i < nFixedCoords; ++i)
        {
            this->m_fullCoordsPtr->col(this->m_varPointIndices(i)) = this->m_curPointsVarPtr->col(i);
        }
        
        m_newError = errorEvaluator.evaluate(*this->m_fullCoordsPtr, *constraintSetPtr, this->m_projectionsPtr);

        // Determine if use local-global result instead of AA. (Move to optimizer.)
        //OMP_SINGLE
        {
            bool requireSwap = (it > 0) && accelerate && (this->m_newError > this->m_curError);
            if (requireSwap)
            {
                std::swap(this->m_curPointsVarPtr, this->m_altPointsVarPtr);
                this->assignInputData(*this->m_curPointsVarPtr);

                if (!hasHandles)
                {
                    this->m_fullCoordsPtr = this->m_curPointsVarPtr;
                }
                newErrorNeedsUpdate = true;
                // If the new error is not smaller than the previous one, re-projection.
                // Else, skip this step. (Move to optimizer.)
                //OMP_FOR
                OMP_PARALLEL_(for)
                for (i32 i = 0; i < nFixedCoords; ++i)
                {
                    this->m_fullCoordsPtr->col(this->m_varPointIndices(i)) = this->m_curPointsVarPtr->col(i);
                }
                this->m_curError = errorEvaluator.evaluate(*this->m_fullCoordsPtr, *constraintSetPtr, this->m_projectionsPtr);
            }
            else
            {
                this->m_curError = this->m_newError;
            }

            this->m_AndersonReset.push_back(requireSwap);
        }

        // Update others.
        //OMP_SINGLE
        {
            // If reach the last iteration, set nRows = 0 to skip the following update.
            if (it == this->m_nIter)
            {
                nColumns = 0;
            }
            else
            {
                std::swap(this->m_prevPointsVarPtr, this->m_curPointsVarPtr);
            }

            //TimerUtil::EventID tIter = this->m_timer.recordTime(("GeometrySolver_Solve_Iter" + std::to_string(it)).c_str());
            //this->m_elapsedTimes.push_back(this->m_timer.getElapsedTime(tBegin, tIter));
            //this->m_funcValues.push_back(curError);
        }

        // Global step: Solve for new points. (Move to optimizer.)
        //OMP_FOR
        OMP_PARALLEL_(for)
        for (i32 i = 0; i < nColumns; ++i)
        {
            VectorX solution;
            solution.resize(this->m_altPointsVarPtr->cols());
            linearSolverPtr->solve(
                this->m_rhsFixedWithInitPoints.row(i).transpose() + this->m_AT * this->m_projectionsPtr->row(i).transpose(), 
                this->m_prevPointsVarPtr->row(i).transpose(), 
                solution);
            this->m_altPointsVarPtr->row(i) = solution.transpose();
        }

        // Anderson acceleration. (Move to optimizer.)
        if (it != this->m_nIter)
        {
            if (accelerate)
            {
                VectorX solution;
                solution.resize(this->m_curPointsVarPtr->size());
                this->m_cur_G = Eigen::Map<const VectorX>(this->m_altPointsVarPtr->data(), this->m_nDimVar);
                this->applyAndersonAcceleration(solution);
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

template<i32 Dim>
inline void AndersonAccelerationOptimizer<Dim>::getOutput(MatrixNX& outQn) const
{
    outQn = *this->m_fullCoordsPtr;
}

template<i32 Dim>
inline void AndersonAccelerationOptimizer<Dim>::applyAndersonAcceleration(VectorX& optimizedQ)
{
    assert(this->m_accumulateIter >= 0);

    this->m_cur_F = this->m_cur_G - this->m_cur_Q;

    if (this->m_accumulateIter == 0)
    {
        this->m_prev_dF.col(0) = -this->m_cur_F;
        this->m_prev_dG.col(0) = -this->m_cur_G;
        this->m_cur_Q = this->m_cur_G;
    }
    else
    {
        this->m_prev_dF.col(this->m_colIdxHistory) += this->m_cur_F;
        this->m_prev_dG.col(this->m_colIdxHistory) += this->m_cur_G;

        scalar eps = glm::epsilon<scalar>();
        scalar scale = glm::max(eps, this->m_prev_dF.col(this->m_colIdxHistory).norm());
        this->m_dF_scale(this->m_colIdxHistory) = scale;
        this->m_prev_dF.col(this->m_colIdxHistory) /= scale;

        i32 mK = glm::min(this->m_mAnderson, this->m_accumulateIter);

        if (mK == 1)
        {
            this->m_theta(0) = 0;
            scalar dF_sqrnorm = this->m_prev_dF.col(this->m_colIdxHistory).squaredNorm();
            this->m_M(0, 0) = dF_sqrnorm;
            scalar dF_norm = glm::sqrt(dF_sqrnorm);

            if (dF_norm > eps)
            {
                this->m_theta(0) = (this->m_prev_dF.col(this->m_colIdxHistory) / dF_norm).dot(this->m_cur_F / dF_norm);
            }
        }
        else
        {
            // Update the normal equation matrix, for the column and row corresponding to the new dF column.
            VectorX newInnerProduction = (this->m_prev_dF.col(this->m_colIdxHistory).transpose() * this->m_prev_dF.block(0, 0, this->m_nDimVar, mK)).transpose();
            this->m_M.block(this->m_colIdxHistory, 0, 1, mK) = newInnerProduction.transpose();
            this->m_M.block(0, this->m_colIdxHistory, mK, 1) = newInnerProduction;

            // Solve normal equation
            this->m_completeOrthoDecomp.compute(this->m_M.block(0, 0, mK, mK));
            this->m_theta.head(mK) = this->m_completeOrthoDecomp.solve(this->m_prev_dF.block(0, 0, this->m_nDimVar, mK).transpose() * this->m_cur_F);
        }

        // Use rescaled theta to compute new U.
        this->m_cur_Q = this->m_cur_G - this->m_prev_dG.block(0, 0, this->m_nDimVar, mK) * ((this->m_theta.head(mK).array() / this->m_dF_scale.head(mK).array()).matrix());

        this->m_colIdxHistory = (this->m_colIdxHistory + 1) % this->m_mAnderson;
        this->m_prev_dF.col(this->m_colIdxHistory) = -this->m_cur_F;
        this->m_prev_dG.col(this->m_colIdxHistory) = -this->m_cur_G;
    }

    ++this->m_accumulateIter;
    optimizedQ = this->m_cur_Q;
}


END_NAMESPACE()
