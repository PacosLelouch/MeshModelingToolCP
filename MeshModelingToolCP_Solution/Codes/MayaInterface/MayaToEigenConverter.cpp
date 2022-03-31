#include "pch.h"
#include <maya/MFloatVectorArray.h>
#include <maya/MFloatPointArray.h>
#include <maya/MFloatArray.h>
#include <maya/MIntArray.h>
#include "MayaToEigenConverter.h"
#include "OpenMPHelper.h"

BEGIN_NAMESPACE(AAShapeUp)

MayaToEigenConverter::MayaToEigenConverter(MObject inMayaMeshObj)
    : m_inMayaMeshObj(inMayaMeshObj)
{
}

void MayaToEigenConverter::setMayaMeshObj(MObject inMayaMeshObj)
{
    m_inMayaMeshObj = inMayaMeshObj;
}

bool MayaToEigenConverter::generateEigenMatrices()
{
    if (!m_inMayaMeshObj.isNull())
    {
        return false;
    }
    MFnMesh fnMesh(m_inMayaMeshObj);
    if (!fnMesh.isValid(MFn::kMesh))
    {
        return false;
    }
    // If a non-api operation happens that many have changed the underlying Maya object wrapped by this api object, make sure that the api object references a valid maya object.
    // In particular this call should be used if you are calling mel commands from your plugin. Note that this only applies for mesh shapes: in a plugin node where the dataMesh is being accessed directly this is not necessary.
    // So is it necessary?
    MStatus status = fnMesh.syncObject();
    CHECK_MSTATUS_AND_RETURN_IT(status);

    m_outMesh.m_section.clear();

    MFloatPointArray positions;
    status = fnMesh.getPoints(positions);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MFloatVectorArray normals;
    status = fnMesh.getVertexNormals(true, normals);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MColorArray colors;
    status = fnMesh.getColors(colors);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MFloatArray us, vs;
    status = fnMesh.getUVs(us, vs);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    m_outMesh.m_positions.resize(Eigen::NoChange, positions.length());
    m_outMesh.m_normals.resize(Eigen::NoChange, normals.length());
    m_outMesh.m_colors.resize(Eigen::NoChange, colors.length());
    m_outMesh.m_texCoords.resize(Eigen::NoChange, us.length());

    for (i64 i = 0; i < m_outMesh.m_positions.cols(); ++i)
    {
        m_outMesh.m_positions.col(i) = toEigenVec3(positions[i]);
    }
    for (i64 i = 0; i < m_outMesh.m_normals.cols(); ++i)
    {
        m_outMesh.m_normals.col(i) = toEigenVec3(normals[i]);
    }
    for (i64 i = 0; i < m_outMesh.m_colors.cols(); ++i)
    {
        m_outMesh.m_colors.col(i) = toEigenVec3(colors[i]);
    }
    for (i64 i = 0; i < m_outMesh.m_texCoords.cols(); ++i)
    {
        m_outMesh.m_texCoords(0, i) = us[i];
        m_outMesh.m_texCoords(1, i) = vs[i];
    }

    i32 numFacesSize = fnMesh.numPolygons(&status);
    m_outMesh.m_section.reserve(numFacesSize * 3ll, numFacesSize);

    for (i32 pid = 0; pid < numFacesSize; ++pid)
    {
        MIntArray vertexList;
        fnMesh.getPolygonVertices(pid, vertexList);

        for (auto& index : vertexList)
        {
            m_outMesh.m_section.m_positionIndices.push_back(index);
            m_outMesh.m_section.m_normalIndices.push_back(index);
            m_outMesh.m_section.m_colorIndices.push_back(index);
            m_outMesh.m_section.m_texCoordsIndices.push_back(index);
        }

        m_outMesh.m_section.m_numFaceVertices.push_back(vertexList.length());
    }
    return true;
}

bool MayaToEigenConverter::updateTargetMesh(MeshDirtyFlag dirtyFlag, MObject outMeshObj, bool updateSurfaceNow)
{
    if (!outMeshObj.isNull())
    {
        return false;
    }
    MFnMesh outFnMesh(outMeshObj);
    if (!outFnMesh.isValid(MFn::kMesh))
    {
        return false;
    }
    MStatus status;

    i64 targetSize = static_cast<i64>(outFnMesh.numVertices(&status));
    CHECK_MSTATUS_AND_RETURN_IT(status);

    if ((dirtyFlag & MeshDirtyFlag::PositionDirty) != MeshDirtyFlag::None)
    {
        OMP_PARALLEL_(for)
        for (i64 i = 0; i < targetSize && i < m_outMesh.m_positions.cols(); ++i)
        {
            MPoint inPoint = fromEigenVec3<MPoint>(m_outMesh.m_positions.col(i));
            outFnMesh.setPoint(i, inPoint);
        }
    }

    if ((dirtyFlag & MeshDirtyFlag::NormalDirty) != MeshDirtyFlag::None)
    {
        OMP_PARALLEL_(for)
        for (i64 i = 0; i < targetSize && i < m_outMesh.m_normals.cols(); ++i)
        {
            MVector inNormal = fromEigenVec3<MVector>(m_outMesh.m_normals.col(i));
            outFnMesh.setVertexNormal(inNormal, i);
        }
    }

    if ((dirtyFlag & MeshDirtyFlag::ColorDirty) != MeshDirtyFlag::None)
    {
        OMP_PARALLEL_(for)
        for (i64 i = 0; i < targetSize && i < m_outMesh.m_colors.cols(); ++i)
        {
            MColor inColor = fromEigenVec3<MColor>(m_outMesh.m_colors.col(i));
            outFnMesh.setColor(i, inColor);
        }
    }

    if ((dirtyFlag & MeshDirtyFlag::TexCoordsDirty) != MeshDirtyFlag::None)
    {
        OMP_PARALLEL_(for)
        for (i64 i = 0; i < targetSize && i < m_outMesh.m_texCoords.cols(); ++i)
        {
            outFnMesh.setUV(i, m_outMesh.m_texCoords(0, i), m_outMesh.m_texCoords(1, i));
        }
    }

    if (updateSurfaceNow && !updateSurface(outFnMesh))
    {
        return false;
    }
    return true;
}

bool MayaToEigenConverter::updateSurface(MFnMesh& outFnMesh)
{
    MStatus status = outFnMesh.syncObject();
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = outFnMesh.updateSurface();
    CHECK_MSTATUS_AND_RETURN_IT(status);
    return true;
}

END_NAMESPACE(AAShapeUp)
