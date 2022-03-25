#pragma once

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
template<i32 Dim>
class ConstraintAbstract
{
public:

    using VectorN = MatrixT<Dim, 1>;
    using MatrixNX = MatrixT<Dim, Eigen::Dynamic>;

    static constexpr i32 getDim() { return Dim; }

    ConstraintAbstract(const std::vector<i32>& idIncidentPoints, scalar weight);

    virtual ~ConstraintAbstract();

    // Set constraint ID. Start constraint ID?
    void setIdConstraint(i32 idConstraint);

    // Get constraint ID. Start constraint ID?
    i32 getIdConstraint() const;

    const VectorXi& getIdIncidentPoints() const;

    scalar getWeight() const;

    // Number of IDs of incident points.
    i32 numIndices() const;

    // Number of the transformed points.
    virtual i32 numTransformedPoints() const = 0;

    // Project, and return the weighted squared distance from the transformed points to the projections.
    virtual scalar project(const MatrixNX& points, MatrixNX& projections) = 0;

    // Extract constraint into triplets of sparse matrix. At the same time, count the constraint ID.
    virtual void extractConstraint(std::vector<SMatrixTriplet>& triplets, i32& inOutConstraintId) = 0;

protected:

    // Generate transform points based on original points. Useless?
    virtual void generateTransformPoints(const MatrixNX& points) = 0;

protected:

    // IDs of incident points of the constraint.
    VectorXi m_idIncidentPoints;

    // The weigth of the constraint.
    scalar m_weight;

protected: // Temporal variables

    // ID of the constraint.
    i32 m_idConstraint;

};




// Interface of constraint component.
template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class IConstraintComponent
{
public:
    USING_MATRIX_VECTOR_SHORTNAME(Dim)
public:
    constexpr IConstraintComponent();

    template<typename TConstraint>
    constexpr void staticTypeCheckBase() const;

    template<typename TConstraint>
    constexpr void staticConvertibleCheckBase() const;

    template<typename TConstraint>
    constexpr TConstraint& staticCast(TConstraintAbstract& constraint) const;

    template<typename TConstraint>
    constexpr TConstraint* staticCast(TConstraintAbstract* constraint) const;
};

// Abstract class of invariant transformer.
template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class InvariantTransformerAbstract : public IConstraintComponent<Dim, TConstraintAbstract> 
{
public:

    // Temporal transformed points.
    typename TConstraintAbstract::MatrixNX m_transformedPoints;
};

// Abstract class of constraint triplet generator.
template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class ConstraintTripletGeneratorAbstract : public IConstraintComponent<Dim, TConstraintAbstract> {};

// Abstract class of constraint projection operator.
template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class ConstraintProjectionOperatorAbstract : public IConstraintComponent<Dim, TConstraintAbstract> {};




// The base class of constraint, with some implementations.
template<i32 Dim, 
    typename TConstraintProjectionOperator = ConstraintProjectionOperatorAbstract<Dim, ConstraintAbstract<Dim> >,
    typename TConstraintTripletGenerator = ConstraintTripletGeneratorAbstract<Dim, ConstraintAbstract<Dim> >, 
    typename TInvariantTransformer = InvariantTransformerAbstract<Dim, ConstraintAbstract<Dim> > >
class ConstraintBase : public ConstraintAbstract<Dim>
{
public:

    using typename ConstraintAbstract<Dim>::VectorN;
    using typename ConstraintAbstract<Dim>::MatrixNX;

    friend typename TConstraintTripletGenerator;
    friend typename TConstraintProjectionOperator;
    friend typename TInvariantTransformer;

    ConstraintBase(const std::vector<i32>& idIncidentPoints, scalar weight);

    virtual ~ConstraintBase();

    virtual i32 numTransformedPoints() const override;

    virtual scalar project(const MatrixNX& points, MatrixNX& projections) override final;

    virtual void extractConstraint(std::vector<SMatrixTriplet>& triplets, i32& inOutConstraintId) override final;

    virtual void generateTransformPoints(const MatrixNX& points) override final;

protected:
    TConstraintProjectionOperator m_projectionOperator;
    TConstraintTripletGenerator m_tripletGenerator;
    TInvariantTransformer m_invariantTransformer;
};


//// Begin some basic constraint components.

// X0, X1, X2 -> X0, X1, X2
template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class IdentityTransformer : public InvariantTransformerAbstract<Dim, TConstraintAbstract>
{
public:
    inline constexpr i32 numTransformedPointsToCreate(const i32 numIndices) const;

    void generateTransformPoints(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& points);
};

// X0, X1, X2 -> X0 - Xc, X1 - Xc, X2 - Xc
template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class MeanCenteringTransformer : public InvariantTransformerAbstract<Dim, TConstraintAbstract>
{
public:
    inline constexpr i32 numTransformedPointsToCreate(const i32 numIndices) const;

    void generateTransformPoints(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& points);
};

// X0, X1, X2 -> 0, X1 - X0, X2 - X0
template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class SubtractFirstTransformer : public InvariantTransformerAbstract<Dim, TConstraintAbstract>
{
public:
    inline constexpr i32 numTransformedPointsToCreate(const i32 numIndices) const;

    void generateTransformPoints(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& points);
};




template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class IdentityWeightTripletGenerator : public ConstraintTripletGeneratorAbstract<Dim, TConstraintAbstract>
{
public:
    void generateTriplets(TConstraintAbstract& constraint, std::vector<SMatrixTriplet>& triplets, i32& inOutConstraintId) const;
};

template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class MeanCenteringWeightTripletGenerator : public ConstraintTripletGeneratorAbstract<Dim, TConstraintAbstract>
{
public:
    void generateTriplets(TConstraintAbstract& constraint, std::vector<SMatrixTriplet>& triplets, i32& inOutConstraintId) const;
};

template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class SubtractFirstWeightTripletGenerator : public ConstraintTripletGeneratorAbstract<Dim, TConstraintAbstract>
{
public:
    void generateTriplets(TConstraintAbstract& constraint, std::vector<SMatrixTriplet>& triplets, i32& inOutConstraintId) const;
};




template<i32 Dim, typename TConstraintAbstract = ConstraintAbstract<Dim> >
class IdentityProjectionOperator : public ConstraintProjectionOperatorAbstract<Dim, TConstraintAbstract>
{
public:
    scalar project(TConstraintAbstract& constraint, const typename TConstraintAbstract::MatrixNX& transformedPoints, typename TConstraintAbstract::MatrixNX& projections) const;
};

//// End some basic constraint components.

//// Start some examples of constraints.

template<i32 Dim>
using IdentityConstraint = ConstraintBase<Dim, IdentityProjectionOperator<Dim>, IdentityWeightTripletGenerator<Dim>, IdentityTransformer<Dim> >;
using IdentityConstraint2D = IdentityConstraint<2>;
using IdentityConstraint3D = IdentityConstraint<3>;

// For extension, put the constraints in an extra class.
template<i32 Dim>
class ConstraintSetAbstract
{
public:
    i32 addConstraint(const std::shared_ptr<ConstraintAbstract<Dim> >& constraintShPtr);

    const std::vector<std::shared_ptr<ConstraintAbstract<Dim> > >& getConstraints() const;

    void clearConstraints();

protected:
    std::vector<std::shared_ptr<ConstraintAbstract<Dim> > > m_constraintShPtrs;
};

template<i32 Dim>
class ConstraintSet : public ConstraintSetAbstract<Dim>
{

};


//// End some examples of constraints.

END_NAMESPACE()

#include "Constraint.inl"
