#include "pch.h"
#include "OperationBase.h"

BEGIN_NAMESPACE(AAShapeUp)

bool OperationBase::initialize(const EigenMesh<3>& mesh, const std::vector<i32>& handleIndices)
{
    //m_vertexIndices = vertexIndices;
    //m_numFaceVertices = numFaceVertices;
    m_initialPositions = mesh.m_positions;
	m_mesh = mesh;
    m_handleIndices = handleIndices;

	m_solverShPtr->clearConstraints();
	m_solverShPtr->clearRegularizations();

    if (!initializeConstraintsAndRegularizations())
    {
        return false;
    }

    return m_solverShPtr && m_solverShPtr->initialize(static_cast<i32>(m_initialPositions.cols()), m_handleIndices);
}

bool OperationBase::solve(Matrix3X& newPositions, i32 nIter)
{
    if (!m_solverShPtr->solve(nIter, &m_initialPositions))
    {
        return false;
    }

    m_solverShPtr->getOutput(newPositions);
    return true;
}

MeshDirtyFlag OperationBase::visualizeOutputErrors(Matrix3X& outColors, scalar maxError, bool keepHeatValue) const
{
	std::vector<scalar> errors;
	auto [dirtyFlag, meshIndexType] = getOutputErrors(errors);

	std::vector<scalar>* errorsPtr = nullptr;
	std::vector<scalar> tempPostProcessErrors;
	switch (meshIndexType)
	{
	case MeshIndexType::PerVertex:
		errorsPtr = &errors;
		break;
	case MeshIndexType::PerTriangle:
		trianglesToVertices(tempPostProcessErrors, errors);
		errorsPtr = &tempPostProcessErrors;
		break;
	case MeshIndexType::PerPolygon:
		polygonsToVertices(tempPostProcessErrors, errors);
		errorsPtr = &tempPostProcessErrors;
		break;
	default:
		return MeshDirtyFlag::None;
	}

	outColors.resize(Eigen::NoChange, static_cast<i64>(errorsPtr->size()));

	if (keepHeatValue)
	{
		OMP_PARALLEL_(for)
		for (i64 col = 0; col < outColors.cols(); ++col)
		{
			Vector3 heatValue = Vector3((*errorsPtr)[col], maxError, scalar(1));
			outColors.col(col) = heatValue;
		}
	}
	else
	{
		OMP_PARALLEL_(for)
		for (i64 col = 0; col < outColors.cols(); ++col)
		{
			Vector3 colorHSV = Vector3(scalar(240) * glm::max(scalar(1) - glm::abs((*errorsPtr)[col]) / glm::abs(maxError), scalar(0)), scalar(1), scalar(1));
			outColors.col(col) = HSV2RGB(colorHSV);
		}
	}
	
	return dirtyFlag;
}

void OperationBase::ReinhardOperatorBatch(Matrix3X& inOutColors)
{
    OMP_PARALLEL_(for)
    for (i64 index = 0; index < inOutColors.size(); ++index)
    {
        inOutColors(index) /= (scalar(1) + inOutColors(index));
    }
}

void OperationBase::HSV2RGBBatch(Matrix3X& inOutColors)
{
    OMP_PARALLEL_(for)
    for (i64 col = 0; col < inOutColors.cols(); ++col)
    {
        Vector3 colorHSV = inOutColors.col(col);
        inOutColors.col(col) = HSV2RGB(colorHSV);
    }
}

Vector3 OperationBase::HSV2RGB(Vector3 inHSV)
{
	scalar minValue;
	scalar chroma;
	scalar hDash;
	scalar x;
	Vector3 RGB;

	chroma = inHSV.y() * inHSV.z();
	hDash = glm::clamp(inHSV.x(), scalar(0), scalar(360)) / scalar(60);
	x = chroma * (scalar(1) - glm::abs(glm::mod(hDash, scalar(2)) - scalar(1)));

	if(hDash < scalar(1))
	{
		RGB.x() = chroma;
		RGB.y() = x;
	}
	else if(hDash < scalar(2))
	{
		RGB.x() = x;
		RGB.y() = chroma;
	}
	else if(hDash < scalar(3))
	{
		RGB.y() = chroma;
		RGB.z() = x;
	}
	else if(hDash < scalar(4))
	{
		RGB.y() = x;
		RGB.z() = chroma;
	}
	else if(hDash < scalar(5))
	{
		RGB.x() = x;
		RGB.z() = chroma;
	}
	else if(hDash <= scalar(6))
	{
		RGB.x() = chroma;
		RGB.z() = x;
	}

	minValue = inHSV.z() - chroma;

	RGB.x() += minValue;
	RGB.y() += minValue;
	RGB.z() += minValue;

	return RGB;
}

void OperationBase::trianglesToVertices(std::vector<scalar>& outValues, const std::vector<scalar>& inValues) const
{
	Matrix3Xi faceVertexIdx;
	m_mesh.m_section.getFaceVertexIndex(faceVertexIdx, false);

	VectorX sumValue;
	VectorX sumArea;
	sumValue.setZero(m_mesh.m_positions.cols());
	sumArea.setZero(m_mesh.m_positions.cols());
	for (i64 i = 0; i < faceVertexIdx.cols(); ++i)
	{
		Vector3i idxs = faceVertexIdx.col(i);
		Vector3 v0 = m_mesh.m_positions.col(idxs(0));
		Vector3 v1 = m_mesh.m_positions.col(idxs(1));
		Vector3 v2 = m_mesh.m_positions.col(idxs(2));

		Vector3 v10 = v1 - v0;
		Vector3 v20 = v2 - v0;

		Vector3 normalFlat = v10.cross(v20);
		scalar doubleArea = normalFlat.norm();
		
		sumArea(idxs(0)) += doubleArea;
		sumArea(idxs(1)) += doubleArea;
		sumArea(idxs(2)) += doubleArea;

		sumValue(idxs(0)) += doubleArea * inValues[i];
		sumValue(idxs(1)) += doubleArea * inValues[i];
		sumValue(idxs(2)) += doubleArea * inValues[i];
	}

	outValues.resize(m_mesh.m_positions.cols());

	OMP_PARALLEL_(for)
	for (i64 i = 0; i < sumValue.size(); ++i)
	{
		outValues[i] = sumValue(i) / sumArea(i);
	}
}

void OperationBase::polygonsToVertices(std::vector<scalar>& outValues, const std::vector<scalar>& inValues) const
{
	VectorX sumValue;
	VectorX sumArea;
	sumValue.setZero(m_mesh.m_positions.cols());
	sumArea.setZero(m_mesh.m_positions.cols());

	size_t startIdx = 0;
	for (size_t i = 0; i < m_mesh.m_section.m_numFaceVertices.size(); ++i)
	{
		i32 numVert = m_mesh.m_section.m_numFaceVertices[i];

		if (numVert < 3)
		{
			continue;
		}
		else
		{
			scalar curSumArea = scalar(0);
			for (i32 j = 1; j + 1 < numVert; ++j)
			{
				Vector3i idxs(
					m_mesh.m_section.m_positionIndices[startIdx + 0],
					m_mesh.m_section.m_positionIndices[startIdx + j],
					m_mesh.m_section.m_positionIndices[startIdx + j + 1]);

				Vector3 v0 = m_mesh.m_positions.col(idxs(0));
				Vector3 v1 = m_mesh.m_positions.col(idxs(1));
				Vector3 v2 = m_mesh.m_positions.col(idxs(2));

				Vector3 v10 = v1 - v0;
				Vector3 v20 = v2 - v0;

				Vector3 normalFlat = v10.cross(v20);
				scalar doubleArea = normalFlat.norm();

				curSumArea += doubleArea;
			}

			for (i32 j = 0; j < numVert; ++j)
			{
				i32 idx = m_mesh.m_section.m_positionIndices[startIdx + j];
				sumArea(idx) += curSumArea;
				sumValue(idx) += curSumArea * inValues[i];
			}
		}
		startIdx += numVert;
	}

	outValues.resize(m_mesh.m_positions.cols());

	OMP_PARALLEL_(for)
	for (i64 i = 0; i < sumValue.size(); ++i)
	{
		outValues[i] = sumValue(i) / sumArea(i);
	}
}

END_NAMESPACE()
