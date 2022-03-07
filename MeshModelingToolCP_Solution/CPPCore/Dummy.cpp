//#include "LGSolver/Constraints/Constraint.h"
//#include "LGSolver/Constraints/EdgeLengthConstraint.h"
#include "LGSolver/Constraints/ConstraintCollection.h"
#include "LGSolver/Solvers/GeometrySolver.h"

BEGIN_NAMESPACE(AAShapeUp)

void DummyTestCompilation()
{
    std::vector<i32> ids({ 0, 1, 2 });
    //ConstraintBase<3, int, int, int> constraint1(ids, 1.0f);
    ConstraintBase<3, IdentityProjectionOperator<3>, IdentityWeightTripletGenerator<3>, IdentityTransformer<3> > constraint3(ids, 1.0f);
    IdentityConstraint3D constraint1(ids, 1.0f);
    EdgeLengthConstraint3D ec(0, 1, 1.0f, 1.0f);

    Matrix3X points, projections;
    ec.project(points, projections);

    GeometrySolver<3> geometrySolver;

    std::unique_ptr<ConstraintAbstract<2> > consUPtr = std::make_unique<EdgeLengthConstraint2D>(0, 1, 1.0f, 1.0f);
    Matrix2X points2, projections2;
    consUPtr->project(points2, projections2);
}

END_NAMESPACE()
