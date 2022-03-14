#pragma once

#include "Solver.h"
#include "SparseMatrixStorage.h"
#include "OpenMPHelper.h"
#include <deque>
#include <unordered_set>

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim,
    typename TTimer = NullTimer,
    typename TOptimizer = LocalGlobalOptimizer<Dim>,
    typename TSPDLinearSolver = Simplicial_LLT_LinearSolver<Dim>,
    typename TRegularizer = LinearRegularizer<Dim>,
    typename TConstraintSet = ConstraintSet<Dim> >
class GeometrySolver : public SolverBase<Dim, 
    TTimer, 
    TOptimizer, 
    TSPDLinearSolver,
    TRegularizer,
    TConstraintSet>
{
public:

    using Super = SolverBase<Dim,
        TTimer, 
        TOptimizer,
        TSPDLinearSolver,
        TRegularizer,
        TConstraintSet>;
    using typename Super::VectorN;
    using typename Super::MatrixNX;
    using typename Super::MatrixXN;

    GeometrySolver();

    virtual ~GeometrySolver();

    virtual bool initialize(i32 nPoints, const std::vector<i32>& handleIndices = std::vector<i32>()) override;

    virtual bool solve(i32 nIter, const MatrixNX* initPointsPtr = nullptr) override;

protected:

    bool isInitialized() const { return m_handleSolverInitialized || m_solverInitialized; }

    scalar evaluateRegularizationTerm(i32 termIdx, const MatrixNX& fullCoords)
    {
        VectorN regTerm = -m_regularizationRhs.row(termIdx).transpose();
        m_regularizationMatrixStorage.evaluate(termIdx, fullCoords, regTerm);
        return regTerm.squaredNorm() * scalar(0.5);
    }

protected:

    MatrixNX m_pointsVar1, m_pointsVar2, m_pointsVar3;

    MatrixNX* m_curPointsVarPtr = nullptr;
    MatrixNX* m_altPointsVarPtr = nullptr;
    MatrixNX* m_prevPointsVarPtr = nullptr;

    // Coordinates of all points. Ping-pong.
    MatrixNX m_fullCoords1, m_fullCoords2;
    MatrixNX* m_fullCoordsPtr = nullptr;

    // Constraint projections. Ping-pong.
    MatrixNX m_projections1, m_projections2;
    MatrixNX* m_projectionsPtr = nullptr;
    
    // Regularization.
    RowSparseStorage m_regularizationMatrixStorage;
    MatrixXN m_regularizationRhs;

    VectorX m_consError;
    VectorX m_regError;
    VectorX m_altConsError;
    VectorX m_altRegError;
    VectorX m_lbfgsError;

    ColMSMatrix m_varSelection, m_handleSelection;
    VectorXi m_handleIndices, m_varPointIndices;
    ColMSMatrix m_rhsHandleContribution;

    ColMSMatrix m_AT;
    MatrixNX m_rhsFixed;

    //RowSparseStorage m_AT_Jacobi;
    //JacobiRowSparseStorage m_globalJacobiStorage;

    bool m_solverInitialized = false;
    bool m_handleSolverInitialized = false;

public:
    std::deque<bool> m_AndersonReset;
    std::vector<double> m_funcValues;
    std::vector<double> m_elapsedTimes;

protected:
    void clearIterationHistory()
    {
        m_AndersonReset.clear();
        m_funcValues.clear();
        m_elapsedTimes.clear();
    }
};

END_NAMESPACE()

#include "GeometrySolver.inl"
