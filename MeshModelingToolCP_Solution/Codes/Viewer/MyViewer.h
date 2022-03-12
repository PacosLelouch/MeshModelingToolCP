#pragma once
#include "viewer.h"
#include "objmodel.h"

class MyViewer : public Viewer
{
public:
	MyViewer(const std::string& name);
	virtual ~MyViewer();
	virtual void createGUIWindow() override;
	virtual void drawScene() override;

	virtual void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) override;
	virtual void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)override;

private:
	int mOperationType = 0;

	float mTimeScale = 1.0f;
	float mTime = 0;
	float mLastTime = 0;	// last gflw time

	bool mLoaded = false;
	std::unique_ptr<ObjModel> mModel;

	float mWeightPlanar, mWeightRef, mWeightFair, mWeight2nd;

	void loadOBJFile();
	void reset();

	float mPickedRayT;	// Store the t of the casted ray when the target is picked
};