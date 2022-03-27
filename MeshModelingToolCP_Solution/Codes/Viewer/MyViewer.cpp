#include "MyViewer.h"
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <filesystem>
#include <nfd.h>
#include "ObjToEigenConverter.h"

#pragma warning( disable : 26812 ) // Enum type warning.

namespace MyViewerOp
{
	enum Op
	{
		Planarization,
		WireMeshDesign,
		ARAP2D,
		TestBoundingSphere,
	};

	const std::vector<const char*> operationTypeNames =
	{
		"Planarization",
		"Wire Mesh Design",
		"ARAP Deformation",
		"Test Bounding Sphere",
	};
}
namespace MyViewerSh
{
	enum Sh
	{
		Model,
		ModelFlat,
		ModelColor,
		ModelNormal,
		ModelNormalFlat,
		ModelWire,
		ModelWireFront,
		ModelHeatValue,
	};
	
	const std::vector<const char*> shadingTypeNames =
	{
		"Full Light",
		"Flat",
		"Color",
		"Normal",
		"Normal Flat",
		"Wire",
		"Wire Front",
		"Heat Value (Error)",
	};
}
namespace MyViewerDisObj
{
	enum DisObj
	{
		ModelProcessed,
		ModelOrigin,
		ModelReference,
	};

	const std::vector<const char*> displayingObjectNames =
	{
		"Processed",
		"Origin",
		"Reference",
	};
}

const std::string MyViewer::noneString = "None";
const std::string MyViewer::sameAsInputString = "Same As Input";

MyViewer::MyViewer(const std::string& name)
	: Viewer(name)
	, mOriginModelText(noneString)
	, mReferenceModelText(sameAsInputString)
	, mModelOrigin(std::make_unique<ObjModel>())
	, mModel(std::make_unique<ObjModel>())
	, mModelReference(std::make_unique<ObjModel>())
	, mGeometrySolverShPtr(std::make_shared<MyGeometrySolver3D>())
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
	ImGui::InputFloat("Model Scale", &mModelScale, 0.01f, 0.2f, "%.3f");
	//ImGui::SliderInt("Num Iteration", &mNumIter, 0, 20);
	ImGui::InputInt("Num Iteration", &mNumIter, 1, 10);
	mNumIter = glm::max(mNumIter, 0);
	ImGui::InputFloat("Max Error Visualization", &mMaxError, 1e-6f, 0.1f, "%.6f");
	mMaxError = glm::max(mMaxError, 1e-6f);

	if (ImGui::RadioButton(MyViewerOp::operationTypeNames[MyViewerOp::Planarization], &mOperationType, MyViewerOp::Planarization)) { resetOperation(); }
	ImGui::SameLine();
	if (ImGui::RadioButton(MyViewerOp::operationTypeNames[MyViewerOp::WireMeshDesign], &mOperationType, MyViewerOp::WireMeshDesign)) { resetOperation(); }
	ImGui::SameLine();
	if (ImGui::RadioButton(MyViewerOp::operationTypeNames[MyViewerOp::ARAP2D], &mOperationType, MyViewerOp::ARAP2D)) { resetOperation(); }
	//ImGui::SameLine();
	if (ImGui::RadioButton(MyViewerOp::operationTypeNames[MyViewerOp::TestBoundingSphere], &mOperationType, MyViewerOp::TestBoundingSphere)) { resetOperation(); }

	if (ImGui::Button("Load Model")) { loadOBJFileToModel(); }
	ImGui::SameLine();
	if (ImGui::Button("Load Reference")) { loadOBJFileToReference(); }
	ImGui::Text("Origin Model: %s", mOriginModelText.c_str());
	ImGui::Text("Reference Model: %s", mReferenceModelText.c_str());

	ImGui::RadioButton(MyViewerDisObj::displayingObjectNames[MyViewerDisObj::ModelProcessed], &mDisplayingObject, MyViewerDisObj::ModelProcessed);
	ImGui::SameLine();
	ImGui::RadioButton(MyViewerDisObj::displayingObjectNames[MyViewerDisObj::ModelOrigin], &mDisplayingObject, MyViewerDisObj::ModelOrigin);
	ImGui::SameLine();
	ImGui::RadioButton(MyViewerDisObj::displayingObjectNames[MyViewerDisObj::ModelReference], &mDisplayingObject, MyViewerDisObj::ModelReference);

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
	case MyViewerSh::ModelHeatValue:
		shaderUsing = mModelHeatValueShader.get();
		break;
	default:
		break;
	}

	drawGridGround(projView);
	if (shaderUsing) {
		shaderUsing->use();
		shaderUsing->setMat4("uProjView", projView);
		shaderUsing->setVec3("uLightPos", glm::vec3(20, 0, 20));
		shaderUsing->setMat4("uModel", model);
		shaderUsing->setMat3("uModelInvTr", glm::mat3(glm::transpose(glm::inverse(model))));
		shaderUsing->setVec3("color", glm::vec3(0.8, 0.4, 0.2));
		shaderUsing->setFloat("uMaxError", mMaxError);

		switch (mDisplayingObject)
		{
		case MyViewerDisObj::ModelProcessed:
			if (mModelLoaded)
			{
				mModel->drawObj();
			}
			break;
		case MyViewerDisObj::ModelOrigin:
			if (mModelLoaded)
			{
				mModelOrigin->drawObj();
			}
			break;
		case MyViewerDisObj::ModelReference:
			if (mReferenceLoaded)
			{
				mModelReference->drawObj();
			}
			else if (mModelLoaded)
			{
				mModelOrigin->drawObj();
			}
			break;
		default:
			break;
		}
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
	if (!mModelLoaded)
	{
		return;
	}

	mPlanarizationOperation->refMesh = mMeshConverterReference.getEigenMesh();
	mPlanarizationOperation->closeness_weight = mPlanarizationParameter.mCloseness;
	mPlanarizationOperation->planarity_weight = mPlanarizationParameter.mPlanarity;
	mPlanarizationOperation->laplacian_weight = mPlanarizationParameter.mLaplacian;
	mPlanarizationOperation->relative_laplacian_weight = mPlanarizationParameter.mRelativeLaplacian;

	auto& mesh = mMeshConverter.getEigenMesh();
	std::cout << "Apply processing " << "\"executePlanarization\"" << "..." << std::endl;

	if (!mPlanarizationOperation->initialize(mesh, {}))
	{
		std::cout << "Fail to initialize!" << std::endl;
		return;
	}

	if (!mPlanarizationOperation->solve(mesh.m_positions, mNumIter))
	{
		std::cout << "Fail to solve!" << std::endl;
		return;
	}

	AAShapeUp::MeshDirtyFlag colorDirtyFlag = mPlanarizationOperation->visualizeOutputErrors(mesh.m_colors, mMaxError, true);
	AAShapeUp::MeshDirtyFlag normalDirtyFlag = AAShapeUp::regenerateNormals(mesh);
	if (!mMeshConverter.updateSourceMesh(mPlanarizationOperation->getMeshDirtyFlag() | colorDirtyFlag | normalDirtyFlag, true))
	{
		std::cout << "Fail to update source mesh!" << std::endl;
		return;
	}
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
	if (!mModelLoaded)
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

	AAShapeUp::MeshDirtyFlag colorDirtyFlag = mTestBoudingSphereOperation->visualizeOutputErrors(mesh.m_colors, mMaxError, true);
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
		ImGui::SliderFloat("Planarity Weight", &mPlanarizationParameter.mPlanarity, 0, 1);
		ImGui::SliderFloat("Closeness Weight", &mPlanarizationParameter.mCloseness, 0, 1);
		ImGui::SliderFloat("Fairness Weight", &mPlanarizationParameter.mLaplacian, 0, 1);
		ImGui::SliderFloat("Relative Fairness Weight", &mPlanarizationParameter.mRelativeLaplacian, 0, 1);
		break;
	case MyViewerOp::WireMeshDesign:

		break;
	case MyViewerOp::ARAP2D:

		break;
	case MyViewerOp::TestBoundingSphere:
		ImGui::SliderFloat("Sphere Projection Weight", &mTestBoundingSphereParameter.mSphereProjection, 0, 1);
		ImGui::SliderFloat("Fairness Weight", &mTestBoundingSphereParameter.mLaplacian, 0, 1);
		break;
	default:
		break;
	}
}

void MyViewer::loadOBJFileToModel()
{
	std::string path = std::filesystem::current_path().parent_path().parent_path().string();
	nfdchar_t* outPath = NULL;
	nfdresult_t result = NFD_OpenDialog("obj", path.c_str(), &outPath);

	if (result == NFD_OKAY && mModelOrigin->loadObj(std::string(outPath))) 
	{
		mOriginModelText = outPath;
		mReferenceModelText = sameAsInputString;
		resetModelToOrigin();
		updateReference(mModelOrigin.get());

		mModelLoaded = true;
		mReferenceLoaded = false;
		resetOperation();
	}
}

void MyViewer::loadOBJFileToReference()
{
	std::string path = std::filesystem::current_path().parent_path().parent_path().string();
	nfdchar_t* outPath = NULL;
	nfdresult_t result = NFD_OpenDialog("obj", path.c_str(), &outPath);

	if (result == NFD_OKAY && mModelReference->loadObj(std::string(outPath))) 
	{
		mReferenceModelText = outPath;
		updateReference(mModelReference.get());

		mReferenceLoaded = true;
		resetOperation();
	}
}

void MyViewer::resetOperation()
{
	switch (mOperationType)
	{
	case MyViewerOp::Planarization:
		mPlanarizationOperation.reset(new AAShapeUp::PlanarizationOperation(mGeometrySolverShPtr));
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
	mMeshConverter.setObjModelPtr(mModel.get());
	mMeshConverter.generateEigenMatrices();
}

void MyViewer::updateReference(ObjModel* objModelPtr)
{
	mMeshConverterReference.setObjModelPtr(objModelPtr);
	mMeshConverterReference.generateEigenMatrices();
}

