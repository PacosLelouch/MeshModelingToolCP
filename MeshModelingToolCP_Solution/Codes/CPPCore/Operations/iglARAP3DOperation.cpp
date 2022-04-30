#include "pch.h"
#include "iglARAP3DOperation.h"

BEGIN_NAMESPACE(AAShapeUp)

bool iglARAP3DOperation::initializeConstraintsAndRegularizations()
{
    //initialize the origin mesh data
    if (arapData.n == 0) {
        originVerts = m_initialPositions;
        m_mesh.m_section.getFaceVertexIndex(faceIndices);
    }

    VectorXi vectorHandle;
    vectorHandle.resize(m_handleIndices.size());
    for (int i = 0; i < m_handleIndices.size(); i++) {
        vectorHandle(i) = m_handleIndices[i];
    }
    if (arapData.n == 0 || lastHandleIndices != vectorHandle) {
        lastHandleIndices = vectorHandle;
        //NOTICE: need to use transpose() and the function can only use double instead of float
        Eigen::MatrixX3d originVertsD = originVerts.cast<double>().transpose();
        MatrixX3i faceIndicesT = faceIndices.transpose();
        return igl::arap_precomputation(originVertsD, faceIndicesT, 3, vectorHandle, arapData);
    }
    return true;
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
    Matrix3X handlePos;
    handlePos.resize(3, m_handleIndices.size());
    for (int i = 0; i < m_handleIndices.size(); i++) {
        handlePos.col(i) = m_initialPositions.col(m_handleIndices[i]);
    }

    //NOTICE: need to use transpose() and the function can only use double instead of float
    Eigen::MatrixX3d handlePosD = handlePos.cast<double>().transpose(), initialPosD = m_initialPositions.cast<double>().transpose();
    bool success = igl::arap_solve(handlePosD, arapData, initialPosD); //TODO: output may be in initialPosD

    newPositions = initialPosD.cast<float>().transpose();
    return success;
}


END_NAMESPACE()
