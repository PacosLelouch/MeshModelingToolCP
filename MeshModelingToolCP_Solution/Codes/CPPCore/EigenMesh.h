#pragma once

#include "TypesCommon.h"
#include <unordered_set>
#include <type_traits>

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

struct EigenEdge
{
    i32 first = -1, second = -1;
    bool operator==(const EigenEdge& edge) const;
};

struct EigenMeshSection
{
public:
    void clear();

    void reserve(size_t idxSize, size_t numFacesSize);

    void getEdgeCountMap(std::unordered_map<EigenEdge, i32>& outEdgeCountMap) const;
    void getBoundaryEdgeSet(std::unordered_set<EigenEdge>& outBoundaryEdgeSet) const;
    void getBoundaryVertexSet(std::unordered_set<i32>& outBoundaryVertexSet) const;

    void getVertexEdgesMap(std::unordered_map<i32, std::unordered_set<EigenEdge>>& outVertexEdgesMap) const;
    void getVertexAdjacentVerticesMap(std::unordered_map<i32, std::unordered_set<i32>>& outVertexAdjacentVerticesMap) const;

    bool getFaceVertexIndex(Matrix3Xi& outFaceVertexIdx, bool mustBeTriangle = false) const;


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
    EigenMeshSection m_section;

    MatrixNX m_positions;
    MatrixNX m_normals;
    MatrixNX m_colors;

    Matrix2X m_texCoords;
};

END_NAMESPACE()

BEGIN_NAMESPACE(std)
template<>
struct hash<AAShapeUp::EigenEdge>
{
    size_t operator()(const AAShapeUp::EigenEdge& k) const;
};
END_NAMESPACE()
