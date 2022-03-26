#include "ObjToEigenConverter.h"
#include "objmodel.h"

BEGIN_NAMESPACE(AAShapeUp)

ObjToEigenConverter::ObjToEigenConverter(ObjModel* objModelPtr)
    : m_inObjModelPtr(objModelPtr)
{
}

bool ObjToEigenConverter::generateEigenMatrices(bool mergeSections)
{
    if (!m_inObjModelPtr)
    {
        return false;
    }

    m_outMesh.m_section.clear();
    
    m_outMesh.m_positions.resize(Eigen::NoChange, m_inObjModelPtr->attrib.vertices.size() / 3);
    m_outMesh.m_normals.resize(Eigen::NoChange, m_inObjModelPtr->attrib.normals.size() / 3);
    m_outMesh.m_colors.resize(Eigen::NoChange, m_inObjModelPtr->attrib.colors.size() / 3);
    m_outMesh.m_texCoords.resize(Eigen::NoChange, m_inObjModelPtr->attrib.texcoords.size() / 2);

    for (i64 i = 0; i < m_outMesh.m_positions.cols(); ++i)
    {
        m_outMesh.m_positions(0, i) = m_inObjModelPtr->attrib.vertices[i * 3 + 0];
        m_outMesh.m_positions(1, i) = m_inObjModelPtr->attrib.vertices[i * 3 + 1];
        m_outMesh.m_positions(2, i) = m_inObjModelPtr->attrib.vertices[i * 3 + 2];
    }
    for (i64 i = 0; i < m_outMesh.m_normals.cols(); ++i)
    {
        m_outMesh.m_normals(0, i) = m_inObjModelPtr->attrib.normals[i * 3 + 0];
        m_outMesh.m_normals(1, i) = m_inObjModelPtr->attrib.normals[i * 3 + 1];
        m_outMesh.m_normals(2, i) = m_inObjModelPtr->attrib.normals[i * 3 + 2];
    }
    for (i64 i = 0; i < m_outMesh.m_colors.cols(); ++i)
    {
        m_outMesh.m_colors(0, i) = m_inObjModelPtr->attrib.colors[i * 3 + 0];
        m_outMesh.m_colors(1, i) = m_inObjModelPtr->attrib.colors[i * 3 + 1];
        m_outMesh.m_colors(2, i) = m_inObjModelPtr->attrib.colors[i * 3 + 2];
    }
    for (i64 i = 0; i < m_outMesh.m_texCoords.cols(); ++i)
    {
        m_outMesh.m_texCoords(0, i) = m_inObjModelPtr->attrib.texcoords[i * 2 + 0];
        m_outMesh.m_texCoords(1, i) = m_inObjModelPtr->attrib.texcoords[i * 2 + 1];
    }

    size_t idxSize = 0;
    size_t numFacesSize = 0;

    for (auto& shape : m_inObjModelPtr->shapes)
    {
        idxSize += shape.mesh.indices.size();
        numFacesSize += shape.mesh.num_face_vertices.size();
    }

    m_outMesh.m_section.reserve(idxSize, numFacesSize);

    for (auto& shape : m_inObjModelPtr->shapes)
    {
        for (auto& index : shape.mesh.indices)
        {
            m_outMesh.m_section.m_positionIndices.push_back(index.vertex_index);
            m_outMesh.m_section.m_normalIndices.push_back(index.normal_index);
            m_outMesh.m_section.m_colorIndices.push_back(index.vertex_index);
            m_outMesh.m_section.m_texCoordsIndices.push_back(index.texcoord_index);
        }
        for (auto& numVertexIndices : shape.mesh.num_face_vertices)
        {
            m_outMesh.m_section.m_numFaceVertices.push_back(numVertexIndices);
        }
    }
    return true;
}

bool ObjToEigenConverter::updateSourceMesh(MeshDirtyFlag dirtyFlag, bool updateBufferNow)
{
    if (!m_inObjModelPtr)
    {
        return false;
    }

    if ((dirtyFlag & MeshDirtyFlag::PositionDirty) != MeshDirtyFlag::None)
    {
        i64 sourceSize = static_cast<i64>(m_inObjModelPtr->attrib.vertices.size() / 3);
        for (i64 i = 0; i < sourceSize && i < m_outMesh.m_positions.cols(); ++i)
        {
            m_inObjModelPtr->attrib.vertices[i * 3 + 0] = m_outMesh.m_positions(0, i);
            m_inObjModelPtr->attrib.vertices[i * 3 + 1] = m_outMesh.m_positions(1, i);
            m_inObjModelPtr->attrib.vertices[i * 3 + 2] = m_outMesh.m_positions(2, i);
        }
    }
    if ((dirtyFlag & MeshDirtyFlag::NormalDirty) != MeshDirtyFlag::None)
    {
        i64 sourceSize = static_cast<i64>(m_inObjModelPtr->attrib.normals.size() / 3);
        for (i64 i = 0; i < sourceSize && i < m_outMesh.m_normals.cols(); ++i)
        {
            m_inObjModelPtr->attrib.normals[i * 3 + 0] = m_outMesh.m_normals(0, i);
            m_inObjModelPtr->attrib.normals[i * 3 + 1] = m_outMesh.m_normals(1, i);
            m_inObjModelPtr->attrib.normals[i * 3 + 2] = m_outMesh.m_normals(2, i);
        }
    }
    if ((dirtyFlag & MeshDirtyFlag::ColorDirty) != MeshDirtyFlag::None)
    {
        i64 sourceSize = static_cast<i64>(m_inObjModelPtr->attrib.colors.size() / 3);
        for (i64 i = 0; i < sourceSize && i < m_outMesh.m_colors.cols(); ++i)
        {
            m_inObjModelPtr->attrib.colors[i * 3 + 0] = m_outMesh.m_colors(0, i);
            m_inObjModelPtr->attrib.colors[i * 3 + 1] = m_outMesh.m_colors(1, i);
            m_inObjModelPtr->attrib.colors[i * 3 + 2] = m_outMesh.m_colors(2, i);
        }
    }
    if ((dirtyFlag & MeshDirtyFlag::TexCoordsDirty) != MeshDirtyFlag::None)
    {
        i64 sourceSize = static_cast<i64>(m_inObjModelPtr->attrib.texcoords.size() / 2);
        for (i64 i = 0; i < sourceSize && i < m_outMesh.m_texCoords.cols(); ++i)
        {
            m_inObjModelPtr->attrib.texcoords[i * 2 + 0] = m_outMesh.m_texCoords(0, i);
            m_inObjModelPtr->attrib.texcoords[i * 2 + 1] = m_outMesh.m_texCoords(1, i);
        }
    }

    if (updateBufferNow)
    {
        updateBuffer();
    }
    return true;
}

void ObjToEigenConverter::updateBuffer()
{
    if (!m_inObjModelPtr)
    {
        return;
    }
    m_inObjModelPtr->generateDrawables();
}


END_NAMESPACE()
