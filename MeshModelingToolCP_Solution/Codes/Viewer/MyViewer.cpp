#include "MyViewer.h"
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <filesystem>
#include <nfd.h>
#include "ObjToEigenConverter.h"

namespace MyViewerOp
{
	constexpr int Planarization = 0;
	constexpr int WireMeshDesign = Planarization + 1;
	constexpr int ARAP2D = WireMeshDesign + 1;
	constexpr int TestBoundingSphere = ARAP2D + 1;
}
namespace MyViewerSh
{
	constexpr int Model = 0;
	constexpr int ModelFlat = Model + 1;
	constexpr int ModelColor = ModelFlat + 1;
	constexpr int ModelNormal = ModelColor + 1;
	constexpr int ModelNormalFlat = ModelNormal + 1;
	constexpr int ModelWire = ModelNormalFlat + 1;
	constexpr int ModelWireFront = ModelWire + 1;
	
	const std::vector<const char*> shadingTypeNames =
	{
		"Model",
		"Model Flat",
		"Model Color",
		"Model Normal",
		"Model Normal Flat",
		"Model Wire",
		"Model Wire Front",
	};
}

MyViewer::MyViewer(const std::string& name)
	: Viewer(name)
	, mModelOrigin(std::make_unique<ObjModel>())
	, mModel(std::make_unique<ObjModel>())
	, mMeshConverter(mModel.get())
	, mGeometrySolverShPtr(std::make_shared<MyGeometrySolver>())
{
	//mModelOrigin = std::make_unique<ObjModel>();
	//mModel = std::make_unique<ObjModel>();
	//mFBXModel.loadFBX("../fbx/BetaCharacter.fbx");
	//mFBXModel.loadShaders();
	////mFBXModel.createShader("../shader/betaCharacter.vert.glsl", "../shader/betaCharacter.frag.glsl");
	//mFBXModel.loadBVHMotion("../motions/Beta/Beta.bvh", true);	// Use it to set the bind matrices

	//std::string path = "../motions/Beta";
	//for (const auto& entry : std::experimental::filesystem::directory_iterator(path))
	//{
	//	std::string extension = entry.path().extension().generic_string();

	//	if (extension.compare(".bvh") == 0)
	//	{
	//		mBVHFilePaths.push_back(entry.path().generic_string());
	//		mBVHFileStems.push_back(entry.path().stem().generic_string());
	//	}
	//}
	//mPickedTarget = nullptr;

	//mMeshConverter = AAShapeUp::ObjToEigenConverter(mModel.get());
	//mMeshConverterShPtr = std::make_shared<AAShapeUp::ObjToEigenConverter>(mModel.get());
}

MyViewer::~MyViewer()
{
}

void MyViewer::createGUIWindow()
{
	ImGui::BeginMainMenuBar();
	ImGui::Combo("Shading Type", &mShadingType, MyViewerSh::shadingTypeNames.data(), static_cast<int>(MyViewerSh::shadingTypeNames.size()), -1);
	ImGui::Text("| App Avg %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::EndMainMenuBar();
	ImGui::Begin("Editor");
	//Viewer::createGUIWindow();
	//ImGui::SliderFloat("Model Scale", &mModelScale, 0.01f, 100.0f);
	ImGui::InputFloat("Model Scale", &mModelScale, 0.01f, 0.2f);
	ImGui::SliderInt("Num Iteration", &mNumIter, 0, 20);

	if (ImGui::RadioButton("Planarization", &mOperationType, MyViewerOp::Planarization)) { resetOperation(); }
	ImGui::SameLine();
	if (ImGui::RadioButton("Wire Mesh Design", &mOperationType, MyViewerOp::WireMeshDesign)) { resetOperation(); }
	ImGui::SameLine();
	if (ImGui::RadioButton("ARAP Deformation", &mOperationType, MyViewerOp::ARAP2D)) { resetOperation(); }
	//ImGui::SameLine();
	if (ImGui::RadioButton("Test Bounding Sphere", &mOperationType, MyViewerOp::TestBoundingSphere)) { resetOperation(); }

	if (ImGui::Button("Load Model")) { loadOBJFile(); }

	createOperationGUI();
	
	if (ImGui::Button("Apply Processing")) 
	{
		std::cout << "Apply processing " << mOperationType << "..." << std::endl;
		switch (mOperationType)
		{
		case MyViewerOp::Planarization:
			executePlanarization();
			break;
		case MyViewerOp::WireMeshDesign:
			executeWireMeshDesign();
			break;
		case MyViewerOp::ARAP2D:
			executeARAP2D();
			break;
		case MyViewerOp::TestBoundingSphere:
			executeTestBoundingSphere();
			break;
		default:
			std::cout << "Nothing happened. Not implemented?" << std::endl;
			break;
		}
	}
	ImGui::SameLine();
	if (ImGui::Button("Reset Model"))
	{
		resetModelToOrigin();
	}
	//if (mFKIKMode == 0)	// FK
	//{
	//	ImGui::SliderFloat("Time Scale", &mTimeScale, 0, 2);
	//	ImGui::Separator();
	//	// List box
	//	ImGui::Text("Motions");
	//	std::vector<const char*> filesChar(mBVHFilePaths.size());
	//	for (int i = 0; i < filesChar.size(); ++i)
	//	{
	//		filesChar[i] = mBVHFileStems[i].c_str();
	//	}
	//	if (ImGui::ListBox("BVH Files", &mCurrentBVHFileIndex, filesChar.data(), mBVHFilePaths.size(), 10))
	//	{
	//		loadBVHFile(mCurrentBVHFileIndex);
	//	}
	//	if (!mLoaded)
	//	{
	//		ImGui::Text("Loading failed!");
	//		ImGui::Text("FBX joints names and BVH joints names are not the same!");
	//	}
	//}
	//else if (mFKIKMode == 1)	// IK
	//{
	//	const char* IKSolvers[] = { "Limb-based", "CCD", "Pseudo Inverse" };
	//	ImGui::Combo("IK Solver", &mIKType, IKSolvers, IM_ARRAYSIZE(IKSolvers));

	//	for (auto& target : mFBXModel.mIKTargets)
	//	{
	//		if (ImGui::DragFloat3(target.jointName.c_str(), target.targetPos))
	//		{
	//			mFBXModel.computeIK(mIKType, target);
	//		}
	//	}
	//}
	ImGui::End();

}

void MyViewer::drawScene()
{
	glEnable(GL_DEPTH_TEST);

	glm::mat4 model = glm::mat4(mModelScale);
	model[3][3] = 1.0f;
	glm::mat4 projView = mCamera.getProjView();
	// Draw Model
	//if (mFKIKMode == 0)
	//{
	//	float currentTime = glfwGetTime();
	//	mFBXModel.updateDeltaT((currentTime - mLastTime) * mTimeScale);
	//	mLastTime = currentTime;
	//}
	//mShowSkeleton ?
	//	mFBXModel.drawSkeleton(projView, model, glm::vec3(1, 1, 1)) :
	//	mFBXModel.drawModel(projView, model, mLightPos, glm::vec3(0.2, 0.9, 1.0));
	//if (mFKIKMode == 1)	// IK
	//{
	//	mFBXModel.drawTargets(projView, model, glm::vec3(0.5, 1.0, 0.4), 20);
	//}
	Shader* shaderUsing = nullptr;
	switch (mShadingType)
	{
	case MyViewerSh::Model:
		shaderUsing = mModelShader.get();
		break;
	case MyViewerSh::ModelFlat:
		shaderUsing = mModelFlatShader.get();
		break;
	case MyViewerSh::ModelColor:
		shaderUsing = mModelColorShader.get();
		break;
	case MyViewerSh::ModelNormal:
		shaderUsing = mModelNormalShader.get();
		break;
	case MyViewerSh::ModelNormalFlat:
		shaderUsing = mModelNormalFlatShader.get();
		break;
	case MyViewerSh::ModelWire:
		shaderUsing = mModelWireShader.get();
		break;
	case MyViewerSh::ModelWireFront:
		shaderUsing = mModelWireFrontShader.get();
		break;
	default:
		break;
	}

	drawGridGround(projView);
	if (mLoaded && shaderUsing) {
		shaderUsing->use();
		shaderUsing->setMat4("uProjView", projView);
		shaderUsing->setVec3("uLightPos", glm::vec3(20, 0, 20));
		shaderUsing->setMat4("uModel", model);
		shaderUsing->setMat3("uModelInvTr", glm::mat3(glm::transpose(glm::inverse(model))));
		shaderUsing->setVec3("color", glm::vec3(0.8, 0.4, 0.2));
		mModel->drawObj();
	}
}

void MyViewer::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	Viewer::mouseButtonCallback(window, button, action, mods);

	//if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) { mPickedTarget = nullptr; }
	//if (mFKIKMode == 0) { return; }

	//// Raycast pick targets
	//if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	//{
	//	double xpos, ypos;
	//	glfwGetCursorPos(window, &xpos, &ypos);
	//	// Normalised Device Coordinates
	//	xpos = (2.0f * xpos) / windowWidth - 1.0f;
	//	ypos = 1.0f - (2.0f * ypos) / windowHeight;
	//	glm::vec4 ray = glm::vec4(xpos, ypos, -1, 1);
	//	ray = glm::inverse(mCamera.getProj()) * ray;
	//	ray = glm::vec4(ray.x, ray.y, -1, 0);
	//	glm::vec3 rayWorld = glm::vec3(glm::inverse(mCamera.getView()) * ray);
	//	rayWorld = glm::normalize(rayWorld);	// Direction of the ray
	//	glm::vec3 origin = mCamera.getEye();
	//	mPickedTarget = nullptr;

	//	for (auto& target : mFBXModel.mIKTargets)
	//	{
	//		float radius = 10;
	//		glm::vec3 center = glm::vec3{ target.targetPos[0], target.targetPos[1], target.targetPos[2] };
	//		float a = glm::dot(rayWorld, rayWorld);
	//		float b = 2 * glm::dot(rayWorld, origin - center);
	//		float c = glm::dot(origin - center, origin - center) - radius * radius;
	//		float discriminant = b * b - 4 * a * c;
	//		if (discriminant >= 0)
	//		{
	//			mPickedTarget = &target;
	//			mPickedRayT = (-b - std::sqrt(discriminant)) / (2 * a);
	//			std::cout << target.jointName << " t:" << mPickedRayT << std::endl;
	//			break;
	//		}
	//	}
	//}
}

void MyViewer::cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
	Viewer::cursorPosCallback(window, xpos, ypos);	// Call parent method

	//if (mPickedTarget && mPickedRayT > 0)
	//{
	//	xpos = (2.0f * xpos) / windowWidth - 1.0f;
	//	ypos = 1.0f - (2.0f * ypos) / windowHeight;
	//	glm::vec4 ray = glm::vec4(xpos, ypos, -1, 1);
	//	ray = glm::inverse(mCamera.getProj()) * ray;
	//	ray = glm::vec4(ray.x, ray.y, -1, 0);
	//	glm::vec3 rayWorld = glm::vec3(glm::inverse(mCamera.getView()) * ray);
	//	rayWorld = glm::normalize(rayWorld);	// Direction of the ray
	//	glm::vec3 origin = mCamera.getEye();
	//	glm::vec3 newPos = origin + mPickedRayT * rayWorld;
	//	mPickedTarget->targetPos[0] = newPos[0];
	//	mPickedTarget->targetPos[1] = newPos[1];
	//	mPickedTarget->targetPos[2] = newPos[2];
	//	mFBXModel.computeIK(mIKType, *mPickedTarget);
	//}

}

void MyViewer::executePlanarization()
{
	std::cout << "Apply processing " << "(TODO)" << std::endl;
}

void MyViewer::executeWireMeshDesign()
{
	std::cout << "Apply processing " << "(TODO)" << std::endl;
}

void MyViewer::executeARAP2D()
{
	std::cout << "Apply processing " << "(TODO)" << std::endl;
}

void MyViewer::executeTestBoundingSphere()
{
	if (!mLoaded)
	{
		return;
	}

	mTestBoudingSphereOperation->m_LaplacianWeight = mTestBoundingSphereParameter.mLaplacian;
	mTestBoudingSphereOperation->m_sphereProjectionWeight = mTestBoundingSphereParameter.mSphereProjection;

	auto& mesh = mMeshConverter.getEigenMesh();
	std::cout << "Apply processing " << "\"executeTestBoundingSphere\"" << "..." << std::endl;
	if (!mTestBoudingSphereOperation->initialize(mesh, {}))
	{
		std::cout << "Fail to initialize!" << std::endl;
		return;
	}

	if (!mTestBoudingSphereOperation->solve(mesh.m_positions, mNumIter))
	{
		std::cout << "Fail to solve!" << std::endl;
		return;
	}

	AAShapeUp::MeshDirtyFlag colorDirtyFlag = mTestBoudingSphereOperation->visualizeOutputErrors(mesh.m_colors, 1.0f);
	AAShapeUp::MeshDirtyFlag normalDirtyFlag = AAShapeUp::regenerateNormals(mesh);
	if (!mMeshConverter.updateSourceMesh(mTestBoudingSphereOperation->getMeshDirtyFlag() | colorDirtyFlag | normalDirtyFlag, true))
	{
		std::cout << "Fail to update source mesh!" << std::endl;
		return;
	}
}

void MyViewer::createOperationGUI()
{
	switch (mOperationType)
	{
	case MyViewerOp::Planarization:
		ImGui::SliderFloat("Planar Weight", &mPlanarizationParameter.mWeightPlanar, 0, 1);
		ImGui::SliderFloat("Ref Weight", &mPlanarizationParameter.mWeightRef, 0, 1);
		ImGui::SliderFloat("Fair Weight", &mPlanarizationParameter.mWeightFair, 0, 1);
		ImGui::SliderFloat("2nd Fair Weight", &mPlanarizationParameter.mWeight2nd, 0, 1);
		break;
	case MyViewerOp::WireMeshDesign:

		break;
	case MyViewerOp::ARAP2D:

		break;
	case MyViewerOp::TestBoundingSphere:
		ImGui::SliderFloat("Sphere Projection", &mTestBoundingSphereParameter.mSphereProjection, 0, 1);
		ImGui::SliderFloat("Laplacian Weight", &mTestBoundingSphereParameter.mLaplacian, 0, 1);
		break;
	default:
		break;
	}
}

void MyViewer::loadOBJFile()
{
	std::string path = std::filesystem::current_path().parent_path().parent_path().string();
	nfdchar_t* outPath = NULL;
	nfdresult_t result = NFD_OpenDialog("obj", path.c_str(), &outPath);

	if (result == NFD_OKAY) {
		mModelOrigin->loadObj(std::string(outPath));
		resetModelToOrigin();
		mLoaded = true;
		resetOperation();
	}
}

void MyViewer::resetOperation()
{
	switch (mOperationType)
	{
	case MyViewerOp::Planarization:

		break;
	case MyViewerOp::WireMeshDesign:

		break;
	case MyViewerOp::ARAP2D:

		break;
	case MyViewerOp::TestBoundingSphere:
		mTestBoudingSphereOperation.reset(new AAShapeUp::TestBoundingSphereOperation(mGeometrySolverShPtr));
		break;
	default:
		std::cout << "Nothing happened. Not implemented?" << std::endl;
		break;
	}
}

void MyViewer::resetModelToOrigin()
{
	mModel->copyObj(*mModelOrigin);
	mMeshConverter.generateEigenMatrices();
}

