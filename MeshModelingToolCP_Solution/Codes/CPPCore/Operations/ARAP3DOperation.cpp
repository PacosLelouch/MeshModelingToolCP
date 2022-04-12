#include "pch.h"
#include "ARAP3DOperation.h"
#include "LGSolver/Constraints/ARAP3DTetConstraint.h"

#include "LGSolver/Regularizations/LaplacianRegTerm.h"
#include <tetgen.h>

BEGIN_NAMESPACE(AAShapeUp)

bool ARAP3DOperation::initializeConstraintsAndRegularizations()
{
    tetgenio input, output;
    char options[10] = "pq";
    m_mesh.toTetgenio(input);
    tetrahedralize(options, &input, &output);
    EigenMesh<3> tmp;
    tmp.fromTetgenio(output);
    m_initialPositions = tmp.m_positions;

    auto& solver = this->m_solverShPtr;
    
    for (int i = 0; i < output.numberoftetrahedra; i++) {
        std::vector<i32> indices(output.tetrahedronlist[i * 4], output.tetrahedronlist[i * 4 + 4]);
        for (auto& n : indices) {
            n -= 1;
        }
        solver->addConstraint(std::make_shared<ARAP3DTetConstraint>(indices, m_deformationWeight, m_initialPositions, true));
    }

    return true;
}

std::tuple<MeshDirtyFlag, MeshIndexType> ARAP3DOperation::getOutputErrors(std::vector<scalar>& outErrors) const
{
    outErrors.resize(m_mesh.m_positions.cols(), 0); //TODO

    return { MeshDirtyFlag::ColorDirty, MeshIndexType::PerVertex };
}

MeshDirtyFlag ARAP3DOperation::getMeshDirtyFlag() const
{
    return MeshDirtyFlag::PositionDirty;
}

bool ARAP3DOperation::solve(Matrix3X& newPositions, i32 nIter)
{
    if (!m_solverShPtr->solve(nIter, &m_initialPositions))
    {
        return false;
    }

    i64 cols = m_mesh.m_positions.cols(); // In case of newPositions is the same as m_mesh.m_positions.
    m_solverShPtr->getOutput(newPositions);
    newPositions.resize(3, cols);
    return true;
}


END_NAMESPACE()
