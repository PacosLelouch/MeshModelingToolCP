#pragma once

#include "glm/glm.hpp"
#include "igl/AABB.h"
#include "TypesCommon.h"
#include <vector>

BEGIN_NAMESPACE(AAShapeUp)

//enum class InvarianceTransformType : ui8
//{
//    IDENTITY, // X0, X1, X2 -> X0, X1, X2
//    MEAN_CENTERING, // X0, X1, X2 -> X0 - Xc, X1 - Xc, X2 - Xc
//    SUBTRACT_FIRST, // X0, X1, X2 -> 0, X1 - X0, X2 - X0
//};
//
//inline constexpr i32 reducePointNumAfterTransform(const InvarianceTransformType type)
//{
//    return type == InvarianceTransformType::SUBTRACT_FIRST ? 1 : 0;
//}

// The abstract class of constraint.
template<i32 N>
class ConstraintAbstract
{
public:

    using VectorN = MatrixT<N, 1>;
    using MatrixNX = MatrixT<N, Eigen::Dynamic>;

    static constexpr i32 getDim() { return N; }

    ConstraintAbstract(const std::vector<i32>& idIncidentPoints, scalar weight);

    virtual ~ConstraintAbstract();

    // Is there any usage?
    void setIdConstraint(i32 idConstraint);

    // Number of IDs of incident points.
    i32 numIndices() const;

    // Number of the transformed points.
    i32 numTransformedPoints() const;

    // Project, and return the weighted squared distance from the transformed points to the projections.
    virtual scalar project(const MatrixNX& points, MatrixNX& projections) = 0;

    // Extract constraint into triplets of sparse matrix, and return the constraint ID.
    virtual i32 extractConstraint(std::vector<SMatrixTriplet>& triplets) = 0;

protected:

    // Generate transform points based on original points.
    virtual void generateTransformPoints(const MatrixNX& points) = 0;

protected:

    // IDs of incident points of the constraint.
    VectorXi m_idIncidentPoints;

    // The weigth of the constraint.
    scalar m_weight;

    // ID of the constraint.
    i32 m_idConstraint;

protected: // Temporal variables

    // Temporal transformed points.
    MatrixNX m_transformedPoints;
};

// Interface of constraint component.
template<i32 N, typename TConstraintAbstract = ConstraintAbstract<N> >
class IConstraintComponent
{
public:
    constexpr IConstraintComponent();

    template<typename TConstraint>
    constexpr void staticCheckBase() const;
};

// Interface of invariant transformer.
template<i32 N, typename TConstraintAbstract = ConstraintAbstract<N> >
class IInvariantTransformer : public IConstraintComponent<N, TConstraintAbstract> {};

// Interface of constraint triplet generator.
template<i32 N, typename TConstraintAbstract = ConstraintAbstract<N> >
class IConstraintTripletGenerator : public IConstraintComponent<N, TConstraintAbstract> {};

// Interface of constraint projection operator.
template<i32 N, typename TConstraintAbstract = ConstraintAbstract<N> >
class IConstraintProjectionOperator : public IConstraintComponent<N, TConstraintAbstract> {};

// The base class of constraint, with some implementations.
template<i32 N,
    typename TInvariantTransformer = IInvariantTransformer<N, ConstraintAbstract<N> >, 
    typename TConstraintTripletGenerator = IConstraintTripletGenerator<N, ConstraintAbstract<N> >, 
    typename TConstraintProjectionOperator = IConstraintProjectionOperator<N, ConstraintAbstract<N> > >
class ConstraintBase : public ConstraintAbstract<N>
{
public:

    using typename ConstraintAbstract<N>::VectorN;
    using typename ConstraintAbstract<N>::MatrixNX;

    friend typename TInvariantTransformer;
    friend typename TConstraintTripletGenerator;
    friend typename TConstraintProjectionOperator;

    ConstraintBase(const std::vector<i32>& idIncidentPoints, scalar weight);

    virtual ~ConstraintBase();

    virtual scalar project(const MatrixNX& points, MatrixNX& projections) override;

    virtual i32 extractConstraint(std::vector<SMatrixTriplet>& triplets) override;

protected:

    virtual void generateTransformPoints(const MatrixNX& points) override;

protected:
    TInvariantTransformer invariantTransformer;
    TConstraintTripletGenerator tripletGenerator;
    TConstraintProjectionOperator projectionOperator;
};

// X0, X1, X2 -> X0, X1, X2
template<i32 N, typename TConstraintAbstract = ConstraintAbstract<N> >
class IdentityTransformer : public IInvariantTransformer<N, TConstraintAbstract>
{
public:
    inline constexpr i32 numTransformedPointsToCreate(const i32 numIndices) const;

    template<typename TConstraint>
    void generateTransformPoints(TConstraint& constraint, const typename TConstraintAbstract::MatrixNX& points) const;
};

// X0, X1, X2 -> X0 - Xc, X1 - Xc, X2 - Xc
template<i32 N, typename TConstraintAbstract = ConstraintAbstract<N> >
class MeanCenteringTransformer : public IInvariantTransformer<N, TConstraintAbstract>
{
public:
    inline constexpr i32 numTransformedPointsToCreate(const i32 numIndices) const;

    template<typename TConstraint>
    void generateTransformPoints(TConstraint& constraint, const typename TConstraintAbstract::MatrixNX& points) const;
};

// X0, X1, X2 -> 0, X1 - X0, X2 - X0
template<i32 N, typename TConstraintAbstract = ConstraintAbstract<N> >
class SubtractFirstTransformer : public IInvariantTransformer<N, TConstraintAbstract>
{
public:
    inline constexpr i32 numTransformedPointsToCreate(const i32 numIndices) const;

    template<typename TConstraint>
    void generateTransformPoints(TConstraint& constraint, const typename TConstraintAbstract::MatrixNX& points) const;
};

END_NAMESPACE()

#include "Constraint.inl"
