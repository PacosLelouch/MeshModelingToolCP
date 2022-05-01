#include "pch.h"
#include "iglARAP3DOperation.h"
#include <igl/arap.h>
#include <tetgen.h>

BEGIN_NAMESPACE(AAShapeUp)

bool iglARAP3DOperation::initializeConstraintsAndRegularizations()
{
#if TET_GEN_FOR_IGL_ARAP3D
    // Begin Tetgen
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
    // End Tetgen

    //initialize the origin mesh data
    if (!arapDataShPtr || arapDataShPtr->n == 0) {
        arapDataShPtr = std::make_shared<igl::ARAPData>();
        arapDataShPtr->energy = igl::ARAP_ENERGY_TYPE_ELEMENTS;
        originVerts = m_initialPositions;

        tetIndices.resize(Eigen::NoChange, m_tempTetMeshIOShPtr->numberoftetrahedra);
        for (int i = 0; i < m_tempTetMeshIOShPtr->numberoftetrahedra; i++) {
            std::vector<i32> indices{ m_tempTetMeshIOShPtr->tetrahedronlist[i * 4], m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 1], m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 2], m_tempTetMeshIOShPtr->tetrahedronlist[i * 4 + 3] };
            for (auto& n : indices) {
                n -= 1;
            }
            tetIndices.col(i) = Vector4i(indices[0], indices[1], indices[2], indices[3]);
        }
    }

    //NOTICE: Eigen matrices cannot use != to check the size.
    if (arapDataShPtr->n == 0 || lastHandleIndices != m_handleIndices) {
        lastHandleIndices = m_handleIndices;

        VectorXi vectorHandle;
        vectorHandle.resize(m_handleIndices.size());
        for (int i = 0; i < m_handleIndices.size(); i++) {
            vectorHandle(i) = m_handleIndices[i];
        }

        //NOTICE: need to use transpose() and the function can only use double instead of float
        Eigen::MatrixX3d originVertsD = originVerts.cast<double>().transpose();
        MatrixX4i tetIndicesT = tetIndices.transpose();
        igl::arap_precomputation(originVertsD, tetIndicesT, 3, vectorHandle, *arapDataShPtr);
        return true;// Ignore numerical issues.
    }
    return true;
#else // !TET_GEN_FOR_IGL_ARAP3D

    //initialize the origin mesh data
    if (!arapDataShPtr || arapDataShPtr->n == 0) {
        arapDataShPtr = std::make_shared<igl::ARAPData>();
        arapDataShPtr->energy = igl::ARAP_ENERGY_TYPE_ELEMENTS;
        originVerts = m_initialPositions;
        if (!m_mesh.m_section.getFaceVertexIndex(faceIndices)) {
            return false;
        }
    }

    //NOTICE: Eigen matrices cannot use != to check the size.
    if (arapDataShPtr->n == 0 || lastHandleIndices != m_handleIndices) {
        lastHandleIndices = m_handleIndices;

        VectorXi vectorHandle;
        vectorHandle.resize(m_handleIndices.size());
        for (int i = 0; i < m_handleIndices.size(); i++) {
            vectorHandle(i) = m_handleIndices[i];
        }

        //NOTICE: need to use transpose() and the function can only use double instead of float
        Eigen::MatrixX3d originVertsD = originVerts.cast<double>().transpose();
        MatrixX3i faceIndicesT = faceIndices.transpose();
        //return igl::arap_precomputation(originVertsD, faceIndicesT, 3, vectorHandle, *arapDataShPtr);
        igl::arap_precomputation(originVertsD, faceIndicesT, 3, vectorHandle, *arapDataShPtr);
        return true;// Ignore numerical issues.
    }
    return true;
#endif // TET_GEN_FOR_IGL_ARAP3D
}

std::tuple<MeshDirtyFlag, MeshIndexType> iglARAP3DOperation::getOutputErrors(std::vector<scalar>& outErrors) const
{
    outErrors.resize(m_mesh.m_positions.cols(), 0); //TODO

    return { MeshDirtyFlag::ColorDirty, MeshIndexType::PerVertex };
}

MeshDirtyFlag iglARAP3DOperation::getMeshDirtyFlag() const
{
    return MeshDirtyFlag::PositionDirty;
}

bool iglARAP3DOperation::solve(Matrix3X& newPositions, i32 nIter)
{
    if (!arapDataShPtr)
    {
        return false;
    }

    Matrix3X handlePos;
    handlePos.resize(3, m_handleIndices.size());
    for (int i = 0; i < m_handleIndices.size(); i++) {
        handlePos.col(i) = m_initialPositions.col(m_handleIndices[i]);
    }

    arapDataShPtr->max_iter = nIter;

    //NOTICE: need to use transpose() and the function can only use double instead of float
    Eigen::MatrixX3d handlePosD = handlePos.cast<double>().transpose(), initialPosD = m_initialPositions.cast<double>().transpose();
    bool success = igl::arap_solve(handlePosD, *arapDataShPtr, initialPosD); //TODO: output may be in initialPosD

    i64 cols = m_mesh.m_positions.cols(); // In case of newPositions is the same as m_mesh.m_positions.
    newPositions = initialPosD.cast<scalar>().transpose();
#if TET_GEN_FOR_IGL_ARAP3D
        newPositions.conservativeResize(3, cols);
#endif // TET_GEN_FOR_IGL_ARAP3D
    return success;
}


END_NAMESPACE()
