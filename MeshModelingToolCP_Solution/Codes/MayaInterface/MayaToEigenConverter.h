#pragma once

#include "TypesCommon.h"
#include "EigenMesh.h"
#include <maya/MFnMesh.h>

BEGIN_NAMESPACE(AAShapeUp)

class MayaToEigenConverter
{
public:
    MayaToEigenConverter(MObject inMayaMeshObj);

    void setMayaMeshObj(MObject inMayaMeshObj);

    bool generateEigenMatrices();

    bool updateTargetMesh(MeshDirtyFlag dirtyFlag, MObject outMeshObj, bool updateSurfaceNow = false);

    bool updateSurface(MFnMesh& outFnMesh);

    EigenMesh<3>& getEigenMesh() { return m_outMesh; }

protected:
    MObject m_inMayaMeshObj;

    EigenMesh<3> m_outMesh;
};

END_NAMESPACE()