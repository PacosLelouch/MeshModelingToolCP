#pragma once

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"

#define BEGIN_NAMESPACE(Namespace) namespace Namespace {
#define END_NAMESPACE() }

BEGIN_NAMESPACE(AAShapeUp)

using i8 = char;
using ui8 = unsigned char;
using i16 = short;
using ui16 = unsigned short;
using i32 = int;
using ui32 = unsigned int;
using i64 = long long;
using ui64 = unsigned long long;
using f32 = float;
using f64 = double;

using scalar = f32;
using ColMSMatrix = Eigen::SparseMatrix<scalar, Eigen::ColMajor>;
using RowMSMatrix = Eigen::SparseMatrix<scalar, Eigen::RowMajor>;
using SMatrixTriplet = Eigen::Triplet<scalar>;

constexpr i32 INVALID_INT = -1;

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
using Matrix3Xi = MatrixTi<3, Eigen::Dynamic>;
using MatrixX3i = MatrixTi<Eigen::Dynamic, 3>;
using Matrix2Xi = MatrixTi<2, Eigen::Dynamic>;
using MatrixX2i = MatrixTi<Eigen::Dynamic, 2>;
using VectorXi = MatrixTi<Eigen::Dynamic, 1>;
using MatrixXXi = MatrixTi<Eigen::Dynamic, Eigen::Dynamic>;

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

END_NAMESPACE()