#include "LGSolver/Constraints/Constraint.h"

BEGIN_NAMESPACE(AAShapeUp)

void DummyTest()
{
    std::vector<i32> ids(1, 0);
    //ConstraintBase<3, int > constraint1(ids, 1.0f);
    ConstraintBase<3, IdentityTransformer<3>, int > constraint3(ids, 1.0f);
}

END_NAMESPACE()
