#pragma once

#include "Constraint.h"
#include "EigenMesh.h"
#include <igl/svd3x3.h>

BEGIN_NAMESPACE(AAShapeUp)

class ARAP3DTetProjectionOperator : public ConstraintProjectionOperatorAbstract<3, ConstraintAbstract<3>>
{
public:
    scalar project(ConstraintAbstract<3>& constraint, const typename ConstraintAbstract<3>::MatrixNX& transformedPoints, typename ConstraintAbstract<3>::MatrixNX& projections) const;

};

class ARAP3DTetTripletGenerator : public ConstraintTripletGeneratorAbstract<3, ConstraintAbstract<3>>
{
public:
    void generateTriplets(ConstraintAbstract<3>& constraint, std::vector<SMatrixTriplet>& triplets, i32& inOutConstraintId) const;
};

class ARAP3DTetTransformer : public InvariantTransformerAbstract<3, ConstraintAbstract<3>> {
public:
    inline constexpr i32 numTransformedPointsToCreate(const i32 numIndices) const { return numIndices; }; //TODO may be -1

    void generateTransformPoints(ConstraintAbstract<3>& constraint, const typename ConstraintAbstract<3>::MatrixNX& points);
};

class ARAP3DTetConstraint : public ConstraintBase<3,
    ARAP3DTetProjectionOperator,
    ARAP3DTetTripletGenerator,
    ARAP3DTetTransformer >
{
public:
    using Super = ConstraintBase<3,
        ARAP3DTetProjectionOperator,
        ARAP3DTetTripletGenerator,
        ARAP3DTetTransformer >;

    ARAP3DTetConstraint(const std::vector<i32> &idI, scalar weight, const Matrix3X& positions, bool fast_svd)
        : Super(idI, weight), use_fast_svd_(fast_svd)
    {
        Matrix34 points;
        for (int i = 0; i < 4; ++i) {
            points.col(i) = positions.col(idI[i]);
        }
        points.colwise() -= points.rowwise().mean().eval();

        Matrix33 edge_vecs;
        for (int i = 0; i < 3; ++i) {
            edge_vecs.col(i) = points.col(i + 1) - points.col(0);
        }

        volume_ = std::fabs(edge_vecs.determinant()) / 6.0;
        m_weight *= std::sqrt(volume_); //TODO

        // Transformation matrix
        transform_ = points.transpose().jacobiSvd(
            Eigen::ComputeFullU | Eigen::ComputeFullV).solve(
                Matrix44::Identity() - Matrix44::Constant(scalar(1) / 4.0)).transpose();
    }

    virtual ~ARAP3DTetConstraint() {}

    typedef Eigen::Matrix<scalar, 4, 3, Eigen::ColMajor | Eigen::DontAlign> UnAlignedMatrix43;
    UnAlignedMatrix43 transform_;  // Transform of the mean-centered vertex positions to compute deformation gradient
    scalar volume_;		// Volume of the original tetrahedron
    bool use_fast_svd_;
};

inline scalar ARAP3DTetProjectionOperator::project(ConstraintAbstract<3>& constraint, const typename ConstraintAbstract<3>::MatrixNX& transformedPoints, typename ConstraintAbstract<3>::MatrixNX& projections) const
{
    using MatrixNX = typename ConstraintAbstract<3>::MatrixNX;
    Eigen::Map<MatrixNX> projectionBlock(&projections(0, constraint.getIdConstraint()), 3, transformedPoints.cols());

    //if (use_fast_svd_) {
        Eigen::Matrix<float, 3, 3> U, V, A, R;

        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                A(i, j) = transformedPoints(i, j);
            }
        }

        Eigen::Matrix<float, 3, 1> S;
        igl::svd3x3(A, U, S, V);
        R = U * V.transpose();

        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                projectionBlock(i, j) = R(i, j);
            }
        }
    //}
    /*else {
        Eigen::JacobiSVD<Matrix33> jSVD(
            M, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Matrix33 I = Matrix33::Identity();
        if ((jSVD.matrixU() * jSVD.matrixV().transpose()).determinant() < 0) {
            I(2, 2) = -1;
        }
        P = jSVD.matrixU() * I * jSVD.matrixV().transpose();
    }*/

    //scalar weighted_dist = (M - P).squaredNorm() * constraint.getWeight() * scalar(0.5);
    //projections.block(0, constraint.getIdConstraint(), 3, 3) = constraint.getWeight() * projectionBlock;
    //return weighted_dist;

    //general code for projection and error
    scalar sqrDist = (transformedPoints - projectionBlock).squaredNorm();
    projectionBlock *= constraint.getWeight(); //TODO
    return sqrDist * (constraint.getWeight()) * static_cast<scalar>(0.5);
    //return sqrDist * (constraint.getWeight() * constraint.getWeight()) * static_cast<scalar>(0.5);
}

inline void ARAP3DTetTripletGenerator::generateTriplets(ConstraintAbstract<3>& constraint, std::vector<SMatrixTriplet>& triplets, i32& inOutConstraintId) const {
    //constraint.m_idConstraint = inOutConstraintId;  TODO

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            triplets.push_back(SMatrixTriplet(inOutConstraintId, constraint.getIdIncidentPoints()[j], 
                sqrt(constraint.getWeight()) * dynamic_cast<ARAP3DTetConstraint*>(&constraint)->transform_(j, i)));
        }

        inOutConstraintId++;
    }
}

inline void ARAP3DTetTransformer::generateTransformPoints(ConstraintAbstract<3>& constraint, const typename ConstraintAbstract<3>::MatrixNX& points) {
    for (i64 i = 0; i < constraint.getIdIncidentPoints().size(); ++i) //TODO = 4
    {
        this->m_transformedPoints.col(i) = points.col(constraint.getIdIncidentPoints()[i]);
    }
    this->m_transformedPoints = this->m_transformedPoints * dynamic_cast<ARAP3DTetConstraint*>(&constraint)->transform_;
}


END_NAMESPACE()
