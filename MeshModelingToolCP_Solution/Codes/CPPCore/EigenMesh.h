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

template<i32 Dim>
struct EigenMeshSection
{
public:
    USING_MATRIX_VECTOR_SHORTNAME(Dim)

public:
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
    std::vector<EigenMeshSection<Dim> > m_sections;

    MatrixNX m_positions;
    MatrixNX m_normals;
    MatrixNX m_colors;

    Matrix2X m_texCoords;
};


END_NAMESPACE()
