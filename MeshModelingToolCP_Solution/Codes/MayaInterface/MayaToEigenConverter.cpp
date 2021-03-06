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

void MayaToEigenConverter::resetOutputEigenMeshToInitial()
{
    m_outMesh = m_initialMesh;
}

bool MayaToEigenConverter::generateEigenMesh()
{
    MStatus status = MStatus::kSuccess;
    if (m_inMayaMeshObj.isNull())
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
    status = fnMesh.syncObject();
    CHECK_MSTATUS_AND_RETURN_IT(status);

    m_outMesh.m_section.clear();

    MFloatPointArray positions;
    status = fnMesh.getPoints(positions);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    ui32 positionLength = positions.length();

    MFloatVectorArray normals;
    status = fnMesh.getVertexNormals(true, normals);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    ui32 normalLength = normals.length();

    MStringArray colorSetNames;
    status = fnMesh.getColorSetNames(colorSetNames);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MColorArray colors;
    if (colorSetNames.length() > 0)
    {
        status = fnMesh.getColors(colors, &colorSetNames[0]);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    ui32 colorLength = colors.length();

    MStringArray uvSetNames;
    status = fnMesh.getUVSetNames(uvSetNames);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MFloatArray us, vs;
    if (uvSetNames.length() > 0)
    {
        status = fnMesh.getUVs(us, vs);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    ui32 uvLength = us.length();

    m_outMesh.m_positions.resize(Eigen::NoChange, positionLength);
    m_outMesh.m_normals.resize(Eigen::NoChange, normalLength);
    m_outMesh.m_colors.resize(Eigen::NoChange, colorLength);
    m_outMesh.m_texCoords.resize(Eigen::NoChange, uvLength);

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
    m_initialMesh = m_outMesh;
    return true;
}

MStatus MayaToEigenConverter::updateTargetMesh(MeshDirtyFlag dirtyFlag, MObject outMeshObj, bool updateSurfaceNow, const MString* colorSet, const MString* uvSet)
{
    MStatus status = MStatus::kSuccess;

    if (!outMeshObj.isNull())
    {
        return MStatus::kFailure;
    }
    MFnMesh outFnMesh(outMeshObj);
    if (!outFnMesh.isValid(MFn::kMesh))
    {
        return MStatus::kFailure;
    }

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
            outFnMesh.setColor(i, inColor, colorSet);
        }
    }

    if ((dirtyFlag & MeshDirtyFlag::TexCoordsDirty) != MeshDirtyFlag::None)
    {
        OMP_PARALLEL_(for)
        for (i64 i = 0; i < targetSize && i < m_outMesh.m_texCoords.cols(); ++i)
        {
            outFnMesh.setUV(i, m_outMesh.m_texCoords(0, i), m_outMesh.m_texCoords(1, i), uvSet);
        }
    }

    if (updateSurfaceNow && !updateSurface(outFnMesh))
    {
        return MStatus::kFailure;
    }
    return status;
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
