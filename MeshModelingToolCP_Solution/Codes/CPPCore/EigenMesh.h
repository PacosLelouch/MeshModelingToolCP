#pragma once

#include "TypesCommon.h"
#include <unordered_set>
#include <type_traits>

class tetgenio;

BEGIN_NAMESPACE(AAShapeUp)

enum class MeshIndexType : ui8
{
    InvalidType,
    PerVertex,
    PerTriangle,
    PerPolygon,
};

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

inline MeshDirtyFlag operator&(const MeshDirtyFlag f1, const MeshDirtyFlag f2)
{
    return MeshDirtyFlag(ui64(f1) & ui64(f2));
}

inline MeshDirtyFlag& operator&=(MeshDirtyFlag& f1, const MeshDirtyFlag f2)
{
    f1 = (f1 & f2);
    return f1;
}

inline MeshDirtyFlag operator|(const MeshDirtyFlag f1, const MeshDirtyFlag f2)
{
    return MeshDirtyFlag(ui64(f1) | ui64(f2));
}

inline MeshDirtyFlag& operator|=(MeshDirtyFlag& f1, const MeshDirtyFlag f2)
{
    f1 = (f1 | f2);
    return f1;
}

struct EigenEdge
{
    i32 first = -1, second = -1;
    bool operator==(const EigenEdge& edge) const;

    i32 getAnotherVertex(i32 curVertex) const;
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

    void toTetgenio(tetgenio& result) const;
    void fromTetgenio(tetgenio& input);

    EigenMeshSection m_section;

    MatrixNX m_positions;
    MatrixNX m_normals;
    MatrixNX m_colors;

    Matrix2X m_texCoords;
};

template<typename TMesh>
MeshDirtyFlag regenerateNormals(TMesh& mesh)
{
    return MeshDirtyFlag::None;
}

template<>
MeshDirtyFlag regenerateNormals(EigenMesh<3>& mesh);

END_NAMESPACE()

BEGIN_NAMESPACE(std)
template<>
struct hash<AAShapeUp::EigenEdge>
{
    size_t operator()(const AAShapeUp::EigenEdge& k) const;
};
END_NAMESPACE()
