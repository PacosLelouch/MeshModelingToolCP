#pragma once

#include "OpenMPHelper.h"
#include "Solver.h"
#include "SparseMatrixStorage.h"

BEGIN_NAMESPACE(AAShapeUp)

template<i32 Dim>
class GeometrySolverErrorEvaluator : public ErrorEvaluatorAbstract<Dim>
{
public:
    using Super = ErrorEvaluatorAbstract<Dim>;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)

    virtual ~GeometrySolverErrorEvaluator() override {}

    virtual scalar evaluate(const MatrixNX& fullQ, ConstraintSetAbstract<Dim>& constraintSet, MatrixNX* outProjectionPtr = nullptr) override;

    VectorX m_consError;
    VectorX m_regError;
};

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
    friend class GeometrySolverErrorEvaluator<Dim>;

    using Super = SolverBase<Dim,
        TTimer, 
        TOptimizer,
        TSPDLinearSolver,
        TRegularizer,
        TConstraintSet>;
    USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(Super)

    GeometrySolver();

    virtual ~GeometrySolver();

    virtual bool initialize(i32 nPoints, const std::vector<i32>& handleIndices = std::vector<i32>()) override;

    virtual bool solve(i32 nIter, const MatrixNX* initPointsPtr = nullptr) override;

    virtual void getOutput(MatrixNX& output) const override;

protected:

    bool isInitialized() const { return this->m_optimizer.isInitialized(); }

protected:

    //MatrixNX m_pointsVar1, m_pointsVar2, m_pointsVar3;

    //MatrixNX* m_curPointsVarPtr = nullptr;
    //MatrixNX* m_altPointsVarPtr = nullptr;
    //MatrixNX* m_prevPointsVarPtr = nullptr;

    //// Coordinates of all points. Ping-pong.
    //MatrixNX m_fullCoords1, m_fullCoords2;
    //MatrixNX* m_fullCoordsPtr = nullptr;

    //// Constraint projections. Ping-pong.
    //MatrixNX m_projections1, m_projections2;
    //MatrixNX* m_projectionsPtr = nullptr;
    
    //// Regularization.
    //RowSparseStorage m_regularizationMatrixStorage;
    //MatrixXN m_regularizationRhs;

    GeometrySolverErrorEvaluator<Dim> m_errorEvaluator;

    //ColMSMatrix m_varSelection, m_handleSelection;
    //VectorXi m_handleIndices, m_varPointIndices;
    //ColMSMatrix m_rhsHandleContribution;

    //ColMSMatrix m_AT;
    //MatrixNX m_rhsFixed;

    //bool m_solverInitialized = false;
    //bool m_handleSolverInitialized = false;

public:
    //std::vector<bool> m_AndersonReset;
    std::vector<double> m_funcValues;
    std::vector<double> m_elapsedTimes;
};

END_NAMESPACE()

#include "GeometrySolver.inl"
