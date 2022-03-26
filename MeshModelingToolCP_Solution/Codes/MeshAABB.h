#pragma once
#include "TypesCommon.h"
#include "CPPCore/EigenMesh.h"
#include <igl/AABB.h>

BEGIN_NAMESPACE(AAShapeUp)

class MeshAABB {
public:
	MeshAABB(const EigenMesh<3>& ref_mesh) {
        Matrix3X ref_mesh_points;
        Eigen::Matrix3Xi ref_mesh_faces;
        ref_mesh_points = ref_mesh.m_positions;
        ref_mesh.m_section.getFaceVertexIndex(ref_mesh_faces);
        vRef = ref_mesh_points.transpose();
        fRef = ref_mesh_faces.transpose();

        AABBTree.init(vRef, fRef);
	};

    scalar getClosestPoint(const Vector3& point, Vector3& closest_point) const {
        int i;
        igl::AABB<MatrixX3, 3>::RowVectorDIMS p = point.transpose(), c;
        scalar sqr_dist = AABBTree.squared_distance(vRef, fRef, p, i, c);
        closest_point = c.transpose();

        return sqr_dist;
    };
private:
	MatrixX3 vRef;
	Eigen::MatrixX3i fRef;
	igl::AABB<MatrixX3, 3> AABBTree;
};

END_NAMESPACE(AAShapeUp)