#pragma once
#include "viewer.h"
#include "objmodel.h"
#include "Operations/PlanarizationOperation.h"
#include "Operations/TestBoundingSphereOperation.h"
#include "Operations/MinimalSurfaceOperation.h"

using MyGeometrySolver3D = AAShapeUp::GeometrySolver<3, AAShapeUp::OpenMPTimer, AAShapeUp::AndersonAccelerationOptimizer<3>, AAShapeUp::Simplicial_LDLT_LinearSolver<3>>;

struct PlanarizationParameter
{
	float mCloseness = 10.0f, mRelativeLaplacian = 10.0f, mLaplacian = 1.0f, mPlanarity = 150.0f;
};

struct TestBoundingSphereParameter
{
	float mSphereProjection = 1.0f, mLaplacian = 0.1f;
};

struct MinimalSurfaceParameter
{
	float mFixedBoundary = 1.0f, mLaplacian = 0.1f;
};

class MyViewer : public Viewer
{
public:
	MyViewer(const std::string& name);
	virtual ~MyViewer();
	virtual void createGUIWindow() override;
	virtual void drawScene() override;

	virtual void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) override;
	virtual void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) override;

	void resetCamera();

	void executePlanarization();
	void executeWireMeshDesign();
	void executeARAP2D();
	void executeTestBoundingSphere();
	void executeMinimalSurface();

protected:
	void createOperationGUI();

protected:
	int mOperationType = 0;
	int mShadingType = 0;
	int mDisplayingObject = 0;

	int mNumIter = 50;

	float mModelScale = 1.0f;
	float mMaxError = 0.05f;

	float mTimeScale = 1.0f;
	float mTime = 0;
	float mLastTime = 0;	// last gflw time

	bool mModelLoaded = false;
	bool mReferenceLoaded = false;
	std::string mOriginModelText;
	std::string mReferenceModelText;

	std::unique_ptr<ObjModel> mModelOrigin;
	std::unique_ptr<ObjModel> mModel;

	std::unique_ptr<ObjModel> mModelReference;

	PlanarizationParameter mPlanarizationParameter;

	TestBoundingSphereParameter mTestBoundingSphereParameter;
	MinimalSurfaceParameter mMinimalSurfaceParameter;


	void loadOBJFileToModel();
	void loadOBJFileToReference();

	void resetOperation();
	void resetModelToOrigin();
	void updateReference(ObjModel* objModelPtr);

	//float mPickedRayT;	// Store the t of the casted ray when the target is picked

	std::shared_ptr<MyGeometrySolver3D> mGeometrySolverShPtr;

	std::unique_ptr<AAShapeUp::PlanarizationOperation> mPlanarizationOperation;

	std::unique_ptr<AAShapeUp::TestBoundingSphereOperation> mTestBoudingSphereOperation;
	std::unique_ptr<AAShapeUp::MinimalSurfaceOperation> mMinimalSurfaceOperation;

	AAShapeUp::ObjToEigenConverter mMeshConverter, mMeshConverterOrigin, mMeshConverterReference;

	static const std::string noneString;
	static const std::string sameAsInputString;
};