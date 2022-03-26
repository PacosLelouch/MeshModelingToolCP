#include "pch.h"
#include "PlanarizationOperation.h"

BEGIN_NAMESPACE(AAShapeUp)

bool PlanarizationOperation::initializeConstraintsAndRegularizations()
{

    Matrix3X p;
    get_vertex_points(this->m_mesh, p);

    GeometrySolver<3> solver;
    std::shared_ptr<TriMeshAABB> aabb = std::make_shared<TriMeshAABB>(ref_mesh);

    if (closeness_weight > 0) {
        int n_vtx = p.cols();
        for (int i = 0; i < n_vtx; ++i) {
            solver.add_constraint(
                new PointToRefSurfaceConstraint(i, closeness_weight, aabb));
        }
        //solver.add_constraint(new ReferenceSurfceConstraint(p.cols(), closeness_weight, ref_mesh_points, ref_mesh_faces));
    }

    // Add closeness constraints
    for (PolyMesh::ConstVertexIter v_it = this->m_mesh.vertices_begin();
        v_it != this->m_mesh.vertices_end(); ++v_it) {
        if (laplacian_weight <= 0 && relative_laplacian_weight <= 0) {
            continue;
        }

        if (!this->m_mesh.is_boundary(*v_it)) {

            std::vector<int> vhs;
            vhs.push_back(v_it->idx());

            for (PolyMesh::ConstVertexVertexIter cvv_it = this->m_mesh.cvv_iter(*v_it);
                cvv_it.is_valid(); ++cvv_it) {
                vhs.push_back(cvv_it->idx());
            }

            if (static_cast<int>(vhs.size()) == 5) {
                if (relative_laplacian_weight > 0) {
                    solver.add_relative_uniform_laplacian(std::vector<int>({ vhs[0],
                        vhs[1], vhs[3] }),
                        relative_laplacian_weight, p);
                    solver.add_relative_uniform_laplacian(std::vector<int>({ vhs[0],
                        vhs[2], vhs[4] }),
                        relative_laplacian_weight, p);
                }

                if (laplacian_weight > 0) {
                    solver.add_uniform_laplacian(std::vector<int>({ vhs[0], vhs[1],
                        vhs[3] }),
                        laplacian_weight);
                    solver.add_uniform_laplacian(std::vector<int>({ vhs[0], vhs[2],
                        vhs[4] }),
                        laplacian_weight);
                }
            }
            else {
                if (relative_laplacian_weight > 0) {
                    solver.add_relative_uniform_laplacian(vhs, relative_laplacian_weight,
                        p);
                }

                if (laplacian_weight > 0) {
                    solver.add_uniform_laplacian(vhs, laplacian_weight);
                }
            }
        }
        else {
            std::vector<int> vhs;
            vhs.push_back(v_it->idx());

            std::vector<int> fhs;
            for (PolyMesh::ConstVertexOHalfedgeIter cvoh_it = this->m_mesh.cvoh_iter(*v_it);
                cvoh_it.is_valid(); ++cvoh_it) {
                if (this->m_mesh.is_boundary(this->m_mesh.edge_handle(*cvoh_it))) {
                    PolyMesh::HalfedgeHandle heh = *cvoh_it;
                    vhs.push_back(this->m_mesh.to_vertex_handle(heh).idx());

                    if (this->m_mesh.is_boundary(heh)) {
                        heh = this->m_mesh.opposite_halfedge_handle(heh);
                    }

                    fhs.push_back(this->m_mesh.face_handle(heh).idx());
                }
            }

            if (static_cast<int>(fhs.size()) == 2 && fhs[0] != fhs[1]) {
                if (relative_laplacian_weight > 0) {
                    solver.add_relative_uniform_laplacian(vhs, relative_laplacian_weight,
                        p);
                }

                if (laplacian_weight > 0) {
                    solver.add_uniform_laplacian(vhs, laplacian_weight);
                }
            }
        }
    }

    for (PolyMesh::ConstFaceIter f_it = this->m_mesh.faces_begin();
        f_it != this->m_mesh.faces_end(); ++f_it) {
        if (planarity_weight > 0) {
            std::vector<int> id_vector;
            for (PolyMesh::ConstFaceVertexIter fv_it = this->m_mesh.cfv_iter(*f_it);
                fv_it.is_valid(); ++fv_it) {
                id_vector.push_back(fv_it->idx());
            }

            if (static_cast<int>(id_vector.size()) > 3) {
                solver.add_constraint(new PlaneConstraint(id_vector, planarity_weight));
            }
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
