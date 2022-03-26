#include "pch.h"
#include "PlanarizationOperation.h"
#include "MeshAABB.h"
#include "LGSolver/Constraints/PointToRefSurfaceConstraint.h"
#include "LGSolver/Constraints/PlaneConstraint.h"
#include "LGSolver/Regularizations/LaplacianRegTerm.h"

BEGIN_NAMESPACE(AAShapeUp)

bool PlanarizationOperation::initializeConstraintsAndRegularizations()
{
    const Matrix3X &vertices = this->m_mesh.m_positions;
    const EigenMeshSection& meshIndices = this->m_mesh.m_section;
    int vertNum = vertices.cols();

    auto& solver = this->m_solverShPtr;
    std::shared_ptr<MeshAABB> refMeshTree = std::make_shared<MeshAABB>(this->refMesh);

    if (closeness_weight > 0) {
        for (int i = 0; i < vertNum; ++i) {
            solver->addConstraint(std::make_shared<PointToRefSurfaceConstraint>(i, closeness_weight, refMeshTree));
        }
    }

    std::unordered_map<i32, std::unordered_set<i32>> vertexAdjacentVerticesMap;
    meshIndices.getVertexAdjacentVerticesMap(vertexAdjacentVerticesMap);
    std::unordered_set<i32> boundaryVertexSet;
    meshIndices.getBoundaryVertexSet(boundaryVertexSet);
    std::unordered_set<EigenEdge> boundaryEdgeSet;
    meshIndices.getBoundaryEdgeSet(boundaryEdgeSet);
    std::unordered_map<i32, std::unordered_set<EigenEdge>> vertexEdgesMap;
    meshIndices.getVertexEdgesMap(vertexEdgesMap);

    for (auto& adjacentVerts : vertexAdjacentVerticesMap) {
        i32 origin = adjacentVerts.first;
        if (boundaryVertexSet.find(origin) == boundaryVertexSet.end()) {
            auto nearSet = adjacentVerts.second;
            if (!nearSet.empty()) {
                std::vector<i32> adjacentVec{ origin };
                adjacentVec.insert(adjacentVec.end(), nearSet.begin(), nearSet.end());

                if (adjacentVec.size() == 5) {
                    std::vector<i32> vector1{ adjacentVec[0], adjacentVec[1], adjacentVec[3] }, vector2{ adjacentVec[0], adjacentVec[2], adjacentVec[4] };
                    if (relative_laplacian_weight > 0) {
                        solver->addRegularizationTerm(std::make_shared<UniformLaplacianRelativeRegTerm<3>>(vector1, relative_laplacian_weight, vertices));
                        solver->addRegularizationTerm(std::make_shared<UniformLaplacianRelativeRegTerm<3>>(vector2, relative_laplacian_weight, vertices));
                    }
                    if (laplacian_weight > 0) {
                        solver->addRegularizationTerm(std::make_shared<UniformLaplacianRegTerm<3>>(vector1, laplacian_weight));
                        solver->addRegularizationTerm(std::make_shared<UniformLaplacianRegTerm<3>>(vector2, laplacian_weight));
                    }
                }
                else {
                    if (relative_laplacian_weight > 0) {
                        solver->addRegularizationTerm(std::make_shared<UniformLaplacianRelativeRegTerm<3>>(adjacentVec, relative_laplacian_weight, vertices));
                    }
                    if (laplacian_weight > 0) {
                        solver->addRegularizationTerm(std::make_shared<UniformLaplacianRegTerm<3>>(adjacentVec, laplacian_weight));
                    }
                }
            }
        }
        else {
            auto& edgeSet = vertexEdgesMap[origin];
            if (edgeSet.size() >= 3) {
                std::vector<i32> adjacentVec{ origin };
                for (auto& e : edgeSet) {
                    if (boundaryEdgeSet.find(e) != boundaryEdgeSet.end()) {
                        adjacentVec.push_back(e.first == origin ? e.second : e.first);
                    }
                }

                if (adjacentVec.size() == 3) {
                    if (relative_laplacian_weight > 0) {
                        solver->addRegularizationTerm(std::make_shared<UniformLaplacianRelativeRegTerm<3>>(adjacentVec, relative_laplacian_weight, vertices));
                    }
                    if (laplacian_weight > 0) {
                        solver->addRegularizationTerm(std::make_shared<UniformLaplacianRegTerm<3>>(adjacentVec, laplacian_weight));
                    }
                }
            }
            
        }
    }

    if (planarity_weight > 0) {
        auto vIter = meshIndices.m_positionIndices.begin();
        for (int vn : meshIndices.m_numFaceVertices) {
            std::vector<i32> indices(vIter, vIter + vn);
            vIter += vn;

            solver->addConstraint(std::make_shared<PlaneConstraint>(indices, planarity_weight));
        }
    }
  
    return true;
}

MeshDirtyFlag PlanarizationOperation::getOutputErrors(std::vector<scalar>& outErrors, scalar maxError) const
{
    //TODO: Generate planarity error as color.
    return MeshDirtyFlag::ColorDirty;
}

MeshDirtyFlag PlanarizationOperation::getMeshDirtyFlag() const
{ 
    return MeshDirtyFlag::PositionDirty; 
}

END_NAMESPACE()
