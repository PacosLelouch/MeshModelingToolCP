#pragma once

#include "TypesCommon.h"
#include "EigenMesh.h"

class ObjModel;

BEGIN_NAMESPACE(AAShapeUp)

class ObjToEigenConverter
{
public:
    ObjToEigenConverter(ObjModel* objModelPtr = nullptr);

    void setObjModelPtr(ObjModel* objModelPtr = nullptr);

    void resetOutputEigenMeshToInitial();

    bool generateEigenMesh();

    bool updateSourceMesh(MeshDirtyFlag dirtyFlag, bool updateBufferNow = false);

    void updateBuffer();

    const EigenMesh<3>& getInitialEigenMesh() { return m_initialMesh; }
    EigenMesh<3>& getOutputEigenMesh() { return m_outMesh; }

protected:
    ObjModel* m_objModelPtr = nullptr;

    EigenMesh<3> m_initialMesh, m_outMesh;
};

END_NAMESPACE()
