#include "ObjToEigenConverter.h"
#include "objmodel.h"

BEGIN_NAMESPACE(AAShapeUp)

ObjToEigenConverter::ObjToEigenConverter(ObjModel* objModelPtr)
    : m_inObjModelPtr(objModelPtr)
{
}

bool ObjToEigenConverter::generateEigenMatrices(bool mergeSections)
{
    m_outMesh.m_sections.clear();
    m_outMesh.m_sections.reserve(m_inObjModelPtr->shapes.size());
    
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

    for (auto& shape : m_inObjModelPtr->shapes)
    {
        auto& newSection = m_outMesh.m_sections.emplace_back();
        newSection.m_positionIndices.reserve(shape.mesh.indices.size());
        newSection.m_normalIndices.reserve(shape.mesh.indices.size());
        newSection.m_colorIndices.reserve(shape.mesh.indices.size());
        newSection.m_texCoordsIndices.reserve(shape.mesh.indices.size());
    
        for (auto& index : shape.mesh.indices)
        {
            newSection.m_positionIndices.push_back(index.vertex_index);
            newSection.m_normalIndices.push_back(index.normal_index);
            newSection.m_colorIndices.push_back(index.vertex_index);
            newSection.m_texCoordsIndices.push_back(index.texcoord_index);
        }
    }
    return true;
}

bool ObjToEigenConverter::updateSourceMesh(MeshDirtyFlag dirtyFlag, bool updateBufferNow)
{
    ui64 dirtyFlagInt = static_cast<ui64>(dirtyFlag);
    if (dirtyFlagInt & static_cast<ui64>(MeshDirtyFlag::PositionDirty))
    {
        i64 sourceSize = static_cast<i64>(m_inObjModelPtr->attrib.vertices.size() / 3);
        for (i64 i = 0; i < sourceSize && i < m_outMesh.m_positions.cols(); ++i)
        {
            m_inObjModelPtr->attrib.vertices[i * 3 + 0] = m_outMesh.m_positions(0, i);
            m_inObjModelPtr->attrib.vertices[i * 3 + 1] = m_outMesh.m_positions(1, i);
            m_inObjModelPtr->attrib.vertices[i * 3 + 2] = m_outMesh.m_positions(2, i);
        }
    }
    if (dirtyFlagInt & static_cast<ui64>(MeshDirtyFlag::NormalDirty))
    {
        i64 sourceSize = static_cast<i64>(m_inObjModelPtr->attrib.normals.size() / 3);
        for (i64 i = 0; i < sourceSize && i < m_outMesh.m_normals.cols(); ++i)
        {
            m_inObjModelPtr->attrib.normals[i * 3 + 0] = m_outMesh.m_normals(0, i);
            m_inObjModelPtr->attrib.normals[i * 3 + 1] = m_outMesh.m_normals(1, i);
            m_inObjModelPtr->attrib.normals[i * 3 + 2] = m_outMesh.m_normals(2, i);
        }
    }
    if (dirtyFlagInt & static_cast<ui64>(MeshDirtyFlag::ColorDirty))
    {
        i64 sourceSize = static_cast<i64>(m_inObjModelPtr->attrib.colors.size() / 3);
        for (i64 i = 0; i < sourceSize && i < m_outMesh.m_colors.cols(); ++i)
        {
            m_inObjModelPtr->attrib.colors[i * 3 + 0] = m_outMesh.m_colors(0, i);
            m_inObjModelPtr->attrib.colors[i * 3 + 1] = m_outMesh.m_colors(1, i);
            m_inObjModelPtr->attrib.colors[i * 3 + 2] = m_outMesh.m_colors(2, i);
        }
    }
    if (dirtyFlagInt & static_cast<ui64>(MeshDirtyFlag::TexCoordsDirty))
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
    m_inObjModelPtr->generateDrawables();
}


END_NAMESPACE()
