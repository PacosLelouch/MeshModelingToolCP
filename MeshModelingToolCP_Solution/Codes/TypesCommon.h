#pragma once

#pragma warning( disable : 4244 ) // Implicit conversion loss of precision.
#pragma warning( disable : 4267 ) // Implicit conversion loss of precision.
//#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"

#include "TypesCommonMinimal.h"

#include <iostream>

BEGIN_NAMESPACE(AAShapeUp)

using ColMSMatrix = Eigen::SparseMatrix<scalar, Eigen::ColMajor>;
using RowMSMatrix = Eigen::SparseMatrix<scalar, Eigen::RowMajor>;
using SMatrixTriplet = Eigen::Triplet<scalar>;

#ifdef EIGEN_DONT_ALIGN
#define EIGEN_ALIGNMENT Eigen::DontAlign
#else
#define EIGEN_ALIGNMENT Eigen::AutoAlign
#endif

template<typename TScalar, int Rows, int Cols, int Options = (Eigen::ColMajor | EIGEN_ALIGNMENT) >
using MatrixTS = Eigen::Matrix<TScalar, Rows, Cols, Options>;

template<int Rows, int Cols, int Options = (Eigen::ColMajor | EIGEN_ALIGNMENT) >
using MatrixT = MatrixTS<scalar, Rows, Cols, Options>;

using Vector2 = MatrixT<2, 1>;
using Matrix22 = MatrixT<2, 2>;
using Matrix23 = MatrixT<2, 3>;
using Vector3 = MatrixT<3, 1>;
using Matrix32 = MatrixT<3, 2>;
using Matrix33 = MatrixT<3, 3>;
using Matrix34 = MatrixT<3, 4>;
using Vector4 = MatrixT<4, 1>;
using Matrix44 = MatrixT<4, 4>;
using Matrix4X = MatrixT<4, Eigen::Dynamic>;
using MatrixX4 = MatrixT<Eigen::Dynamic, 4>;
using Matrix3X = MatrixT<3, Eigen::Dynamic>;
using MatrixX3 = MatrixT<Eigen::Dynamic, 3>;
using Matrix2X = MatrixT<2, Eigen::Dynamic>;
using MatrixX2 = MatrixT<Eigen::Dynamic, 2>;
using VectorX = MatrixT<Eigen::Dynamic, 1>;
using MatrixXX = MatrixT<Eigen::Dynamic, Eigen::Dynamic>;

template<int Rows, int Cols, int Options = (Eigen::ColMajor | EIGEN_ALIGNMENT) >
using MatrixTi = MatrixTS<i32, Rows, Cols, Options>;

using Vector2i = MatrixTi<2, 1>;
using Matrix22i = MatrixTi<2, 2>;
using Matrix23i = MatrixTi<2, 3>;
using Vector3i = MatrixTi<3, 1>;
using Matrix32i = MatrixTi<3, 2>;
using Matrix33i = MatrixTi<3, 3>;
using Matrix34i = MatrixTi<3, 4>;
using Vector4i = MatrixTi<4, 1>;
using Matrix44i = MatrixTi<4, 4>;
using Matrix4Xi = MatrixTi<4, Eigen::Dynamic>;
using MatrixX4i = MatrixTi<Eigen::Dynamic, 4>;
using Matrix3Xi = MatrixTi<3, Eigen::Dynamic>;
using MatrixX3i = MatrixTi<Eigen::Dynamic, 3>;
using Matrix2Xi = MatrixTi<2, Eigen::Dynamic>;
using MatrixX2i = MatrixTi<Eigen::Dynamic, 2>;
using VectorXi = MatrixTi<Eigen::Dynamic, 1>;
using MatrixXXi = MatrixTi<Eigen::Dynamic, Eigen::Dynamic>;

template<i32 Dim, typename TVec>
inline MatrixT<Dim, 1> toEigenVec(const TVec& vec)
{
    MatrixT<Dim, 1> outVec;
    for (i64 i = 0; i < Dim; ++i)
    {
        outVec(i) = vec[i];
    }
    return outVec;
}

template<i32 Dim, typename TVec>
inline TVec fromEigenVec(const MatrixT<Dim, 1> & vec)
{
    TVec outVec;
    for (i64 i = 0; i < Dim; ++i)
    {
        outVec[i] = vec(i);
    }
    return outVec;
}

template<typename TVec>
inline Vector3 toEigenVec3(const TVec& vec)
{
    return Vector3(vec[0], vec[1], vec[2]);
}

template<typename TVec>
inline TVec fromEigenVec3(const Vector3& vec)
{
    TVec v;
    v[0] = vec(0);
    v[1] = vec(1);
    v[2] = vec(2);
    return v;
}

#define USING_MATRIX_VECTOR_SHORTNAME(Dim) \
    using VectorN = MatrixT<Dim, 1>; \
    using MatrixNX = MatrixT<Dim, Eigen::Dynamic>; \
    using MatrixXN = MatrixT<Eigen::Dynamic, Dim>;

#define USING_SUPER_CLASS_MATRIX_VECTOR_SHORTNAME(SuperClass) \
    using typename SuperClass::VectorN; \
    using typename SuperClass::MatrixNX; \
    using typename SuperClass::MatrixXN;

END_NAMESPACE()
