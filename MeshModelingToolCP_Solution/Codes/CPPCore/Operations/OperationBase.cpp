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

MeshDirtyFlag OperationBase::visualizeOutputErrors(Matrix3X& outColors, scalar maxError) const
{
	std::vector<scalar> errors;
	MeshDirtyFlag dirtyFlag = getOutputErrors(errors, maxError);

	outColors.resize(Eigen::NoChange, static_cast<i64>(errors.size()));

	OMP_PARALLEL_(for)
	for (i64 col = 0; col < outColors.cols(); ++col)
	{
		Vector3 colorHSV = Vector3(scalar(240) * glm::max(scalar(1) - glm::abs(errors[col]) / glm::abs(maxError), scalar(0)), scalar(1), scalar(1));
		outColors.col(col) = HSV2RGB(colorHSV);
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

END_NAMESPACE()
