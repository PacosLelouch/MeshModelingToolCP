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
            if (indices.size() > 3)
            {
                solver->addConstraint(std::make_shared<PlaneConstraint>(indices, planarity_weight));
            }
        }
    }
  
    return true;
}

std::tuple<MeshDirtyFlag, MeshIndexType> PlanarizationOperation::getOutputErrors(std::vector<scalar>& outErrors) const
{
    Matrix3X finalPositions;
    this->m_solverShPtr->getOutput(finalPositions);
    outErrors.resize(this->m_mesh.m_section.m_numFaceVertices.size());

    auto vIter = this->m_mesh.m_section.m_positionIndices.begin();
    int n = 0;
    for (i32 vn : this->m_mesh.m_section.m_numFaceVertices) {
        Matrix3X facePoints;
        facePoints.resize(Eigen::NoChange, vn);
        for (int i = 0; i < vn; i++) {
            facePoints.col(i) = finalPositions.col(*(vIter + i));
        }
        vIter += vn;

        Eigen::JacobiSVD<Matrix3X> jSVD;
        jSVD.compute(facePoints, Eigen::ComputeFullU);
        Vector3 bestFitNormal = jSVD.matrixU().col(2).normalized();
        Matrix3X projectionBlock = facePoints - bestFitNormal * (bestFitNormal.transpose() * facePoints);
        scalar sqrDist = (facePoints - projectionBlock).squaredNorm();
        outErrors[n] = sqrDist * 20;
        n++;
    }

    return { MeshDirtyFlag::ColorDirty, MeshIndexType::PerPolygon };
}

MeshDirtyFlag PlanarizationOperation::getMeshDirtyFlag() const
{ 
    return MeshDirtyFlag::PositionDirty; 
}

END_NAMESPACE()
