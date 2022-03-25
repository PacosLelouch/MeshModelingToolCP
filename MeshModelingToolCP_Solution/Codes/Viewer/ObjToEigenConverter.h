#pragma once

#include "TypesCommon.h"
#include "EigenMesh.h"

class ObjModel;

BEGIN_NAMESPACE(AAShapeUp)

class ObjToEigenConverter
{
public:
    ObjToEigenConverter(ObjModel* objModelPtr = nullptr);

    bool generateEigenMatrices(bool mergeSections = false);

    bool updateSourceMesh(MeshDirtyFlag dirtyFlag, bool updateBufferNow = false);

    void updateBuffer();

    EigenMesh<3>& getEigenMesh() { return m_outMesh; }

protected:
    ObjModel* m_inObjModelPtr = nullptr;

    EigenMesh<3> m_outMesh;
};

END_NAMESPACE()
