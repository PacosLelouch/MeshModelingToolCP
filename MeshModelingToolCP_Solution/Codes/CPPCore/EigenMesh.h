#pragma once

#include "TypesCommon.h"

BEGIN_NAMESPACE(AAShapeUp)

enum class MeshDirtyFlag : ui64
{
    // Single bit.
    None = 0ll,
    PositionDirty = 1ll,
    NormalDirty = 1ll << 1,
    ColorDirty = 1ll << 2,

    TexCoordsDirty = 1ll << 3,

    // Combination.
    AllDirty = PositionDirty | NormalDirty | ColorDirty | TexCoordsDirty,

    // End bit.
    EndFlag = AllDirty + 1ll,
};

struct EigenMeshSection
{
public:
    void clear()
    {
        m_positionIndices.clear();
        m_normalIndices.clear();
        m_colorIndices.clear();
        m_texCoordsIndices.clear();

        m_numFaceVertices.clear();
    }

    void reserve(size_t idxSize, size_t numFacesSize)
    {
        m_positionIndices.reserve(idxSize);
        m_normalIndices.reserve(idxSize);
        m_colorIndices.reserve(idxSize);
        m_texCoordsIndices.reserve(idxSize);

        m_numFaceVertices.reserve(numFacesSize);
    }

    std::vector<i32> m_positionIndices;
    std::vector<i32> m_normalIndices;
    std::vector<i32> m_colorIndices;

    std::vector<i32> m_texCoordsIndices;

    std::vector<i32> m_numFaceVertices;
};

template<i32 Dim>
struct EigenMesh
{
public:
    USING_MATRIX_VECTOR_SHORTNAME(Dim)

public:
    void getTriangleVertexIndex(Eigen::Matrix3Xi& result) const {
        int faceNum = m_section.m_positionIndices.size() / 3;
        result.resize(3, faceNum);
        for (int i = 0; i < faceNum; i++) {
            result.col(i) = { m_section.m_positionIndices[i * 3], m_section.m_positionIndices[i * 3 + 1], m_section.m_positionIndices[i * 3 + 2] };
        }
    };

    EigenMeshSection m_section;

    MatrixNX m_positions;
    MatrixNX m_normals;
    MatrixNX m_colors;

    Matrix2X m_texCoords;
};


END_NAMESPACE()
