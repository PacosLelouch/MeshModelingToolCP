#include "pch.h"
#include "ARAP3DOperation.h"
#include "LGSolver/Constraints/ARAP3DTetConstraint.h"

#include "LGSolver/Regularizations/LaplacianRegTerm.h"
#include <tetgen.h>

BEGIN_NAMESPACE(AAShapeUp)

bool ARAP3DOperation::initializeConstraintsAndRegularizations()
{
    if (!m_usingCache || !m_tempTetMeshIOShPtr || !m_tempTetEigenMeshShPtr)
    {
        m_tempTetMeshIOShPtr = std::make_shared<tetgenio>();
        m_tempTetEigenMeshShPtr = std::make_shared<EigenMesh<3> >();
        tetgenio input;
        char options[10] = "pq";
        m_mesh.toTetgenio(input);
        tetrahedralize(options, &input, m_tempTetMeshIOShPtr.get());
        m_tempTetEigenMeshShPtr->fromTetgenio(*m_tempTetMeshIOShPtr.get());
    }
    m_usingCache = false;
    
    std::unordered_set<i32> handleIndiceSet(m_handleIndices.begin(), m_handleIndices.end());
    m_initialPositions.conservativeResize(Eigen::NoChange, glm::max(m_initialPositions.cols(), m_tempTetEigenMeshShPtr->m_positions.cols()));
    for (i64 i = 0; i < m_tempTetEigenMeshShPtr->m_positions.cols(); ++i)
    {
        if (handleIndiceSet.find(i) != handleIndiceSet.end())
        {
            continue;
        }
        m_initialPositions.col(i) = m_tempTetEigenMeshShPtr->m_positions.col(i);
    }

    auto& solver = this->m_solverShPtr;

    //std::vector<int> temp_tetrahedronlist;
    //for (int i = 0; i < m_tempTetMeshIOShPtr->numberoftetrahedra; i++) {
    //    temp_tetrahedronlist.push_back(m_tempTetMeshIOShPtr->tetrahedronlist[i * 4]);
    //    temp_tetrahedronlist.push_back(m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 1]);
    //    temp_tetrahedronlist.push_back(m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 2]);
    //    temp_tetrahedronlist.push_back(m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 3]);
    //    std::cout << m_tempTetMeshIOShPtr->tetrahedronlist[i * 4] << ' ' << m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 1] << ' ' << m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 2] << ' ' << m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 3] << std::endl;
    //}

    for (int i = 0; i < m_tempTetMeshIOShPtr->numberoftetrahedra; i++) {

        //std::vector<i32> indices(&m_tempTetMeshIOShPtr->tetrahedronlist[i * 4], &m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 4]);
        std::vector<i32> indices{ m_tempTetMeshIOShPtr->tetrahedronlist[i * 4], m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 1], m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 2], m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 3] };
        for (auto& n : indices) {
            n -= 1;
        }
        solver->addConstraint(std::make_shared<ARAP3DTetConstraint>(indices, m_deformationWeight, m_tempTetEigenMeshShPtr->m_positions, true));
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
    newPositions.conservativeResize(3, cols);
    return true;
}


END_NAMESPACE()
