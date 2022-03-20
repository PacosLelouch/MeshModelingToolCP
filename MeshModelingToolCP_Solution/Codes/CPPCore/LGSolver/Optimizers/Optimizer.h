#pragma once

#include "TypesCommon.h"
#include "SparseMatrixStorage.h"
#include "OpenMPHelper.h"
#include "LGSolver/LinearSolvers/SPDLinearSolver.h"
#include <algorithm>
#include <vector>
#include <unordered_set>

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
class ConstraintSetAbstract;

template<i32 Dim>
class ErrorEvaluatorAbstract
{
public:
    USING_MATRIX_VECTOR_SHORTNAME(Dim)

    virtual ~ErrorEvaluatorAbstract() {}

    virtual scalar evaluate(const MatrixNX& fullQ, ConstraintSetAbstract<Dim>& constraintSet, MatrixNX* outProjectionPtr = nullptr) = 0;

    scalar evaluateRegularizationTerm(i32 termIdx, const MatrixNX& fullCoords) const
    {
        VectorN regTerm = -this->m_regularizationRhs.row(termIdx).transpose();
        this->m_regularizationMatrixStorage.evaluate(termIdx, fullCoords, regTerm);
        return regTerm.squaredNorm() * scalar(0.5);
    }

    // Regularization.
    RowSparseStorage m_regularizationMatrixStorage;
    MatrixXN m_regularizationRhs;
};

template<i32 Dim>
class OptimizerAbstract
{
public:
    USING_MATRIX_VECTOR_SHORTNAME(Dim)

    virtual ~OptimizerAbstract() {}

    virtual bool initialize(i32 nPoints, ConstraintSetAbstract<Dim>& constraintSet, RegularizerAbstract<Dim>& regularizer,
        ErrorEvaluatorAbstract<Dim>* errorEvaluatorPtr = nullptr, SPDLinearSolverAbstract<Dim>* linearSolverPtr = nullptr, const std::vector<i32>& handleIndices = std::vector<i32>()) = 0;

    virtual bool isInitialized() const = 0;

    virtual bool preBeginOptimization(i32 nIter, const MatrixNX* initPointsPtr = nullptr) = 0;

    virtual void assignInputData(const MatrixNX& inQ) = 0;

    virtual bool optimize(ErrorEvaluatorAbstract<Dim>& errorEvaluator, ConstraintSetAbstract<Dim>* constraintSetPtr = nullptr, SPDLinearSolverAbstract<Dim>* linearSolverPtr = nullptr) = 0;

    virtual void getOutput(MatrixNX& outQn) const = 0;

    virtual scalar getCurrentError() const = 0;

protected:

    i32 m_nIter = 0;
};

template<i32 Dim>
class LocalGlobalOptimizer : public OptimizerAbstract<Dim>
{
public:
    using Super = OptimizerAbstract<Dim>;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)

    virtual bool initialize(i32 nPoints, ConstraintSetAbstract<Dim>& constraintSet, RegularizerAbstract<Dim>& regularizer, 
        ErrorEvaluatorAbstract<Dim>* errorEvaluatorPtr = nullptr, SPDLinearSolverAbstract<Dim>* linearSolverPtr = nullptr, const std::vector<i32>& handleIndices = std::vector<i32>()) override;

    virtual bool isInitialized() const override;

    virtual bool preBeginOptimization(i32 nIter, const MatrixNX* initPointsPtr = nullptr) override;

    virtual void assignInputData(const MatrixNX& inQ) override;

    virtual bool optimize(ErrorEvaluatorAbstract<Dim>& errorEvaluator, ConstraintSetAbstract<Dim>* constraintSetPtr = nullptr, SPDLinearSolverAbstract<Dim>* linearSolverPtr = nullptr) override;

    virtual void getOutput(MatrixNX& outQn) const override;
    
    virtual scalar getCurrentError() const override { return this->m_curError; }

    //void assignG(const MatrixNX& g);

protected:
    scalar m_curError = scalar(0);

    // Points for computing.
    MatrixNX m_pointsVar1;
    MatrixNX* m_curPointsVarPtr = nullptr;

    // Coordinates of all points.
    MatrixNX m_fullCoords1;
    MatrixNX* m_fullCoordsPtr = nullptr;

    // Constraint projections.
    MatrixNX m_projections1;
    MatrixNX* m_projectionsPtr = nullptr;

    // Handles.
    ColMSMatrix m_varSelection, m_handleSelection;
    VectorXi m_handleIndices, m_varPointIndices;
    ColMSMatrix m_rhsHandleContribution;

    // Right hand side.
    ColMSMatrix m_AT;
    MatrixNX m_rhsFixed;

    // Flags for initialization.
    bool m_solverInitialized = false;
    bool m_handleSolverInitialized = false;

protected:

    VectorX m_cur_Q;

    VectorX m_cur_G;

    i32 m_nDimVar;
    i32 m_accumulateIter;
    i32 m_colIdxHistory;
};

template<i32 Dim>
class AndersonAccelerationOptimizer : public LocalGlobalOptimizer<Dim>
{
public:
    using Super = LocalGlobalOptimizer<Dim>;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)

    virtual bool initialize(i32 nPoints, ConstraintSetAbstract<Dim>& constraintSet, RegularizerAbstract<Dim>& regularizer,
        ErrorEvaluatorAbstract<Dim>* errorEvaluatorPtr = nullptr, SPDLinearSolverAbstract<Dim>* linearSolverPtr = nullptr, const std::vector<i32>& handleIndices = std::vector<i32>()) override;

    virtual bool isInitialized() const override;

    virtual bool preBeginOptimization(i32 nIter, const MatrixNX* initPointsPtr = nullptr) override;

    virtual bool optimize(ErrorEvaluatorAbstract<Dim>& errorEvaluator, ConstraintSetAbstract<Dim>* constraintSetPtr = nullptr, SPDLinearSolverAbstract<Dim>* linearSolverPtr = nullptr) override;

    virtual void getOutput(MatrixNX& outQn) const override;

    void applyAndersonAcceleration(VectorX& optimizedQ);

    void setNumberOfHistoryUsed(i32 m) { m_mAnderson = m; }
    i32 getNumberOfHistoryUsed() const { return m_mAnderson; }

protected:
    // Additional points for computing. Ping-pong.
    MatrixNX m_pointsVar2, m_pointsVar3;
    MatrixNX* m_altPointsVarPtr = nullptr;
    MatrixNX* m_prevPointsVarPtr = nullptr;

    // Additional coordinates of all points. Ping-pong.
    MatrixNX m_fullCoords2;

    std::vector<bool> m_AndersonReset;
    i32 m_mAnderson = 5;

protected:
    scalar m_newError = scalar(0);
    MatrixNX m_rhsFixedWithInitPoints;

    
    VectorX m_cur_F;
    MatrixXX m_prev_dG;
    MatrixXX m_prev_dF;
    MatrixXX m_M;
    VectorX m_theta;
    VectorX m_dF_scale;

    Eigen::CompleteOrthogonalDecomposition<MatrixXX> m_completeOrthoDecomp;
};

END_NAMESPACE()

#include "Optimizer.inl"
