#include "pch.h"
#include "EigenMesh.h"

BEGIN_NAMESPACE(AAShapeUp)

bool EigenEdge::operator==(const EigenEdge& edge) const
{
    return (first == edge.first && second == edge.second) || (first == edge.second && second == edge.first);
}

i32 EigenEdge::getAnotherVertex(i32 curVertex) const
{
    if (curVertex == first)
    {
        return second;
    }
    else if (curVertex == second)
    {
        return first;
    }
    return INVALID_INT;
}

void EigenMeshSection::clear()
{
    m_positionIndices.clear();
    m_normalIndices.clear();
    m_colorIndices.clear();
    m_texCoordsIndices.clear();

    m_numFaceVertices.clear();
}

void EigenMeshSection::reserve(size_t idxSize, size_t numFacesSize)
{
    m_positionIndices.reserve(idxSize);
    m_normalIndices.reserve(idxSize);
    m_colorIndices.reserve(idxSize);
    m_texCoordsIndices.reserve(idxSize);

    m_numFaceVertices.reserve(numFacesSize);
}

void EigenMeshSection::getEdgeCountMap(std::unordered_map<EigenEdge, i32>& outEdgeCountMap) const
{
    size_t startIdx = 0;
    outEdgeCountMap.clear();
    for (size_t i = 0; i < m_numFaceVertices.size(); ++i)
    {
        i32 numVert = m_numFaceVertices[i];

        for (i32 i = 0; i + 1 < numVert; ++i)
        {
            EigenEdge newEdge{ m_positionIndices[startIdx + i], m_positionIndices[startIdx + i + 1] };
            auto it = outEdgeCountMap.find(newEdge);
            if (it == outEdgeCountMap.end())
            {
                outEdgeCountMap.insert(std::make_pair(newEdge, 1));
            }
            else
            {
                ++it->second;
            }
        }
        EigenEdge newEdge{ m_positionIndices[startIdx + numVert - 1], m_positionIndices[startIdx] };
        auto it = outEdgeCountMap.find(newEdge);
        if (it == outEdgeCountMap.end())
        {
            outEdgeCountMap.insert(std::make_pair(newEdge, 1));
        }
        else
        {
            ++it->second;
        }

        startIdx += numVert;
    }
}

void EigenMeshSection::getBoundaryEdgeSet(std::unordered_set<EigenEdge>& outBoundaryEdgeSet) const
{
    std::unordered_map<EigenEdge, i32> edgeCountMap;
    getEdgeCountMap(edgeCountMap);
    outBoundaryEdgeSet.clear();
    for (auto& edgeCountPair : edgeCountMap)
    {
        if (edgeCountPair.second == 1)
        {
            outBoundaryEdgeSet.insert(edgeCountPair.first);
        }
    }
}

void EigenMeshSection::getBoundaryVertexSet(std::unordered_set<i32>& outBoundaryVertexSet) const
{
    std::unordered_set<EigenEdge> boundaryEdgeSet;
    getBoundaryEdgeSet(boundaryEdgeSet);
    outBoundaryVertexSet.clear();
    for (auto& edge : boundaryEdgeSet)
    {
        outBoundaryVertexSet.insert(edge.first);
        outBoundaryVertexSet.insert(edge.second);
    }
}

void EigenMeshSection::getVertexEdgesMap(std::unordered_map<i32, std::unordered_set<EigenEdge>>& outVertexEdgesMap) const
{
    std::unordered_map<EigenEdge, i32> edgeCountMap;
    getEdgeCountMap(edgeCountMap);
    outVertexEdgesMap.clear();
    for (auto& edgeCountPair : edgeCountMap)
    {
        auto& edge = edgeCountPair.first;

        auto it1 = outVertexEdgesMap.find(edge.first);
        if (it1 == outVertexEdgesMap.end())
        {
            outVertexEdgesMap.insert(std::make_pair(edge.first, std::unordered_set<EigenEdge>{edge}));
        }
        else
        {
            it1->second.insert(edge);
        }

        auto it2 = outVertexEdgesMap.find(edge.second);
        if (it2 == outVertexEdgesMap.end())
        {
            outVertexEdgesMap.insert(std::make_pair(edge.second, std::unordered_set<EigenEdge>{edge}));
        }
        else
        {
            it2->second.insert(edge);
        }
    }
}

void EigenMeshSection::getVertexAdjacentVerticesMap(std::unordered_map<i32, std::unordered_set<i32>>& outVertexAdjacentVerticesMap) const
{
    std::unordered_map<EigenEdge, i32> edgeCountMap;
    getEdgeCountMap(edgeCountMap);
    outVertexAdjacentVerticesMap.clear();
    for (auto& edgeCountPair : edgeCountMap)
    {
        auto& edge = edgeCountPair.first;

        auto it1 = outVertexAdjacentVerticesMap.find(edge.first);
        if (it1 == outVertexAdjacentVerticesMap.end())
        {
            outVertexAdjacentVerticesMap.insert(std::make_pair(edge.first, std::unordered_set<i32>{edge.second}));
        }
        else
        {
            it1->second.insert(edge.second);
        }

        auto it2 = outVertexAdjacentVerticesMap.find(edge.second);
        if (it2 == outVertexAdjacentVerticesMap.end())
        {
            outVertexAdjacentVerticesMap.insert(std::make_pair(edge.second, std::unordered_set<i32>{edge.first}));
        }
        else
        {
            it2->second.insert(edge.first);
        }
    }
}

bool EigenMeshSection::getFaceVertexIndex(Matrix3Xi& outFaceVertexIdx, bool mustBeTriangle) const
{
    std::vector<std::tuple<i32, i32, i32>> triVertIdxs;
    size_t startIdx = 0;
    for (size_t i = 0; i < m_numFaceVertices.size(); ++i)
    {
        i32 numVert = m_numFaceVertices[i];
        
        if (mustBeTriangle && numVert != 3)
        {
            return false;
        }
        
        if (numVert < 3)
        {
            continue;
        }
        else
        {
            for (i32 j = 1; j + 1 < numVert; ++j)
            {
                triVertIdxs.push_back(std::make_tuple(
                    m_positionIndices[startIdx + 0],
                    m_positionIndices[startIdx + j],
                    m_positionIndices[startIdx + j + 1]));
            }
        }
        startIdx += numVert;
    }

    outFaceVertexIdx.resize(Eigen::NoChange, triVertIdxs.size());
    for (i64 i = 0; i < i64(triVertIdxs.size()); ++i)
    {
        auto& triTuple = triVertIdxs[i];
        outFaceVertexIdx.col(i) = Vector3i(std::get<0>(triTuple), std::get<1>(triTuple), std::get<2>(triTuple));//Eigen::Map<Vector3i>(&std::get<0>(triTuple), 3);
    }
    return true;
}

template<>
MeshDirtyFlag regenerateNormals(EigenMesh<3>& mesh)
{
    Matrix3Xi faceVertexIdx;
    if (!mesh.m_section.getFaceVertexIndex(faceVertexIdx, false))
    {
        return MeshDirtyFlag::None;
    }

    mesh.m_normals.setZero(mesh.m_normals.rows(), mesh.m_positions.cols());

    for (i64 i = 0; i < faceVertexIdx.cols(); ++i)
    {
        Vector3i idxs = faceVertexIdx.col(i);
        Vector3 v0 = mesh.m_positions.col(idxs(0));
        Vector3 v1 = mesh.m_positions.col(idxs(1));
        Vector3 v2 = mesh.m_positions.col(idxs(2));

        Vector3 v10 = v1 - v0;
        Vector3 v20 = v2 - v0;

        Vector3 normalFlat = v10.cross(v20);
        mesh.m_normals.col(idxs(0)) += normalFlat;
        mesh.m_normals.col(idxs(1)) += normalFlat;
        mesh.m_normals.col(idxs(2)) += normalFlat;
    }

    for (i64 i = 0; i < mesh.m_normals.cols(); ++i)
    {
        mesh.m_normals.col(i).normalize();
    }

    return MeshDirtyFlag::NormalDirty;
}

END_NAMESPACE()

BEGIN_NAMESPACE(std)

inline size_t hash<AAShapeUp::EigenEdge>::operator()(const AAShapeUp::EigenEdge& k) const
{
    return size_t(k.first) * size_t(k.second);
}

END_NAMESPACE()
