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

    void resetOutputEigenMeshToInitial();

    bool generateEigenMesh();

    MStatus updateTargetMesh(MeshDirtyFlag dirtyFlag, MObject outMeshObj, bool updateSurfaceNow = false, const MString* colorSet = nullptr, const MString* uvSet = nullptr);

    bool updateSurface(MFnMesh& outFnMesh);

    const EigenMesh<3>& getInitialEigenMesh() { return m_initialMesh; }
    EigenMesh<3>& getOutputEigenMesh() { return m_outMesh; }

protected:
    MObject m_inMayaMeshObj;

    EigenMesh<3> m_initialMesh, m_outMesh;
};

END_NAMESPACE()
