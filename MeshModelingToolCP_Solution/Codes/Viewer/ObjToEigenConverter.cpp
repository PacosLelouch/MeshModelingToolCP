#include "ObjToEigenConverter.h"
#include "objmodel.h"
#include "OpenMPHelper.h"

BEGIN_NAMESPACE(AAShapeUp)

ObjToEigenConverter::ObjToEigenConverter(ObjModel* objModelPtr)
    : m_objModelPtr(objModelPtr)
{
}

void ObjToEigenConverter::setObjModelPtr(ObjModel* objModelPtr)
{
    m_objModelPtr = objModelPtr;
}

void ObjToEigenConverter::resetOutputEigenMeshToInitial()
{
    m_outMesh = m_initialMesh;
}

bool ObjToEigenConverter::generateEigenMesh()
{
    if (!m_objModelPtr)
    {
        return false;
    }

    m_outMesh.m_section.clear();
    
    m_outMesh.m_positions.resize(Eigen::NoChange, m_objModelPtr->attrib.vertices.size() / 3);
    m_outMesh.m_normals.resize(Eigen::NoChange, m_objModelPtr->attrib.normals.size() / 3);
    m_outMesh.m_colors.resize(Eigen::NoChange, m_objModelPtr->attrib.colors.size() / 3);
    m_outMesh.m_texCoords.resize(Eigen::NoChange, m_objModelPtr->attrib.texcoords.size() / 2);

    for (i64 i = 0; i < m_outMesh.m_positions.cols(); ++i)
    {
        m_outMesh.m_positions(0, i) = m_objModelPtr->attrib.vertices[i * 3 + 0];
        m_outMesh.m_positions(1, i) = m_objModelPtr->attrib.vertices[i * 3 + 1];
        m_outMesh.m_positions(2, i) = m_objModelPtr->attrib.vertices[i * 3 + 2];
    }
    for (i64 i = 0; i < m_outMesh.m_normals.cols(); ++i)
    {
        m_outMesh.m_normals(0, i) = m_objModelPtr->attrib.normals[i * 3 + 0];
        m_outMesh.m_normals(1, i) = m_objModelPtr->attrib.normals[i * 3 + 1];
        m_outMesh.m_normals(2, i) = m_objModelPtr->attrib.normals[i * 3 + 2];
    }
    for (i64 i = 0; i < m_outMesh.m_colors.cols(); ++i)
    {
        m_outMesh.m_colors(0, i) = m_objModelPtr->attrib.colors[i * 3 + 0];
        m_outMesh.m_colors(1, i) = m_objModelPtr->attrib.colors[i * 3 + 1];
        m_outMesh.m_colors(2, i) = m_objModelPtr->attrib.colors[i * 3 + 2];
    }
    for (i64 i = 0; i < m_outMesh.m_texCoords.cols(); ++i)
    {
        m_outMesh.m_texCoords(0, i) = m_objModelPtr->attrib.texcoords[i * 2 + 0];
        m_outMesh.m_texCoords(1, i) = m_objModelPtr->attrib.texcoords[i * 2 + 1];
    }

    size_t idxSize = 0;
    size_t numFacesSize = 0;

    for (auto& shape : m_objModelPtr->shapes)
    {
        idxSize += shape.mesh.indices.size();
        numFacesSize += shape.mesh.num_face_vertices.size();
    }

    m_outMesh.m_section.reserve(idxSize, numFacesSize);

    for (auto& shape : m_objModelPtr->shapes)
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
    m_initialMesh = m_outMesh;
    return true;
}

bool ObjToEigenConverter::updateSourceMesh(MeshDirtyFlag dirtyFlag, bool updateBufferNow)
{
    if (!m_objModelPtr)
    {
        return false;
    }

    if ((dirtyFlag & MeshDirtyFlag::PositionDirty) != MeshDirtyFlag::None)
    {
        i64 sourceSize = static_cast<i64>(m_objModelPtr->attrib.vertices.size() / 3);

        OMP_PARALLEL_(for)
        for (i64 i = 0; i < sourceSize && i < m_outMesh.m_positions.cols(); ++i)
        {
            m_objModelPtr->attrib.vertices[i * 3 + 0] = m_outMesh.m_positions(0, i);
            m_objModelPtr->attrib.vertices[i * 3 + 1] = m_outMesh.m_positions(1, i);
            m_objModelPtr->attrib.vertices[i * 3 + 2] = m_outMesh.m_positions(2, i);
        }
    }
    if ((dirtyFlag & MeshDirtyFlag::NormalDirty) != MeshDirtyFlag::None)
    {
        if (m_objModelPtr->attrib.normals.size() != m_outMesh.m_normals.size() * 3)
        {
            m_objModelPtr->attrib.normals.resize(m_outMesh.m_normals.size() * 3);
            for (auto& shape : m_objModelPtr->shapes)
            {
                for (auto& index : shape.mesh.indices)
                {
                    index.normal_index = index.vertex_index;
                }
            }
        }
        i64 sourceSize = static_cast<i64>(m_objModelPtr->attrib.normals.size() / 3);

        OMP_PARALLEL_(for)
        for (i64 i = 0; i < sourceSize && i < m_outMesh.m_normals.cols(); ++i)
        {
            m_objModelPtr->attrib.normals[i * 3 + 0] = m_outMesh.m_normals(0, i);
            m_objModelPtr->attrib.normals[i * 3 + 1] = m_outMesh.m_normals(1, i);
            m_objModelPtr->attrib.normals[i * 3 + 2] = m_outMesh.m_normals(2, i);
        }
    }
    if ((dirtyFlag & MeshDirtyFlag::ColorDirty) != MeshDirtyFlag::None)
    {
        if (m_objModelPtr->attrib.colors.size() != m_outMesh.m_colors.size() * 3)
        {
            m_objModelPtr->attrib.colors.resize(m_outMesh.m_colors.size() * 3);
        }
        i64 sourceSize = static_cast<i64>(m_objModelPtr->attrib.colors.size() / 3);

        OMP_PARALLEL_(for)
        for (i64 i = 0; i < sourceSize && i < m_outMesh.m_colors.cols(); ++i)
        {
            m_objModelPtr->attrib.colors[i * 3 + 0] = m_outMesh.m_colors(0, i);
            m_objModelPtr->attrib.colors[i * 3 + 1] = m_outMesh.m_colors(1, i);
            m_objModelPtr->attrib.colors[i * 3 + 2] = m_outMesh.m_colors(2, i);
        }
    }
    if ((dirtyFlag & MeshDirtyFlag::TexCoordsDirty) != MeshDirtyFlag::None)
    {
        if (m_objModelPtr->attrib.texcoords.size() != m_outMesh.m_texCoords.size() * 3)
        {
            m_objModelPtr->attrib.texcoords.resize(m_outMesh.m_texCoords.size() * 3);
            for (auto& shape : m_objModelPtr->shapes)
            {
                for (auto& index : shape.mesh.indices)
                {
                    index.texcoord_index = index.vertex_index;
                }
            }
        }
        i64 sourceSize = static_cast<i64>(m_objModelPtr->attrib.texcoords.size() / 2);

        OMP_PARALLEL_(for)
        for (i64 i = 0; i < sourceSize && i < m_outMesh.m_texCoords.cols(); ++i)
        {
            m_objModelPtr->attrib.texcoords[i * 2 + 0] = m_outMesh.m_texCoords(0, i);
            m_objModelPtr->attrib.texcoords[i * 2 + 1] = m_outMesh.m_texCoords(1, i);
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
    if (!m_objModelPtr)
    {
        return;
    }
    m_objModelPtr->generateDrawables();
}


END_NAMESPACE()
