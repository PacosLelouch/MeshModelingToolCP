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
		MinimalSurface,
	};

	const std::vector<const char*> operationTypeNames =
	{
		"Planarization",
		"Wire Mesh Design",
		"ARAP Deformation",
		"Test Bounding Sphere",
		"Minimal Surface",
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
	ImGui::SameLine();
	if (ImGui::RadioButton(MyViewerOp::operationTypeNames[MyViewerOp::MinimalSurface], &mOperationType, MyViewerOp::MinimalSurface)) { resetOperation(); }

	if (ImGui::Button("Load Model")) { loadOBJFileToModel(); }
	ImGui::SameLine();
	if (ImGui::Button("Load Reference")) { loadOBJFileToReference(); }
	ImGui::SameLine();
	if (ImGui::Button("Reset Camera")) { resetCamera(); }
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
		case MyViewerOp::MinimalSurface:
			executeMinimalSurface();
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
	
	ImGui::End();

}

void MyViewer::drawScene()
{
	glEnable(GL_DEPTH_TEST);

	glm::mat4 model = glm::mat4(mModelScale);
	model[3][3] = 1.0f;
	glm::mat4 projView = mCamera.getProjView();
	
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

	
}

void MyViewer::cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
	Viewer::cursorPosCallback(window, xpos, ypos);	// Call parent method

}

void MyViewer::resetCamera()
{
	mCamera = Camera(windowWidth, windowHeight, glm::vec3(0, 4, 8), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
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

	if (!mPlanarizationOperation->solve(mesh.m_positions, 0) ||
		!mMeshConverterOrigin.updateSourceMesh(mPlanarizationOperation->visualizeOutputErrors(mMeshConverterOrigin.getEigenMesh().m_colors, mMaxError, true), true))
	{
		std::cout << "Fail to get visualized error!" << std::endl;
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

	if (!mTestBoudingSphereOperation->solve(mesh.m_positions, 0) || 
		!mMeshConverterOrigin.updateSourceMesh(mTestBoudingSphereOperation->visualizeOutputErrors(mMeshConverterOrigin.getEigenMesh().m_colors, mMaxError, true), true))
	{
		std::cout << "Fail to get visualized error!" << std::endl;
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

void MyViewer::executeMinimalSurface()
{
	if (!mModelLoaded)
	{
		return;
	}

	mMinimalSurfaceOperation->m_LaplacianWeight = mMinimalSurfaceParameter.mLaplacian;
	mMinimalSurfaceOperation->m_fixBoundaryWeight = mMinimalSurfaceParameter.mFixedBoundary;

	auto& mesh = mMeshConverter.getEigenMesh();
	std::cout << "Apply processing " << "\"executeMinimalSurface\"" << "..." << std::endl;
	if (!mMinimalSurfaceOperation->initialize(mesh, {}))
	{
		std::cout << "Fail to initialize!" << std::endl;
		return;
	}

	if (!mMinimalSurfaceOperation->solve(mesh.m_positions, 0) || 
		!mMeshConverterOrigin.updateSourceMesh(mMinimalSurfaceOperation->visualizeOutputErrors(mMeshConverterOrigin.getEigenMesh().m_colors, mMaxError, true), true))
	{
		std::cout << "Fail to get visualized error!" << std::endl;
		return;
	}

	if (!mMinimalSurfaceOperation->solve(mesh.m_positions, mNumIter))
	{
		std::cout << "Fail to solve!" << std::endl;
		return;
	}

	AAShapeUp::MeshDirtyFlag colorDirtyFlag = mMinimalSurfaceOperation->visualizeOutputErrors(mesh.m_colors, mMaxError, true);
	AAShapeUp::MeshDirtyFlag normalDirtyFlag = AAShapeUp::regenerateNormals(mesh);
	if (!mMeshConverter.updateSourceMesh(mMinimalSurfaceOperation->getMeshDirtyFlag() | colorDirtyFlag | normalDirtyFlag, true))
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
		ImGui::InputFloat("Planarity Weight", &mPlanarizationParameter.mPlanarity, 0.1f, 10.0f);
		ImGui::InputFloat("Closeness Weight", &mPlanarizationParameter.mCloseness, 0.1f, 10.0f);
		ImGui::InputFloat("Fairness Weight", &mPlanarizationParameter.mLaplacian, 0.1f, 10.0f);
		ImGui::InputFloat("Relative Fairness Weight", &mPlanarizationParameter.mRelativeLaplacian, 0.1f, 10.0f);
		break;
	case MyViewerOp::WireMeshDesign:

		break;
	case MyViewerOp::ARAP2D:

		break;
	case MyViewerOp::TestBoundingSphere:
		ImGui::InputFloat("Sphere Projection Weight", &mTestBoundingSphereParameter.mSphereProjection, 0.1f, 10.0f);
		ImGui::InputFloat("Fairness Weight", &mTestBoundingSphereParameter.mLaplacian, 0.1f, 10.0f);
		break;
	case MyViewerOp::MinimalSurface:
		ImGui::InputFloat("Fixed Boundary Weight", &mMinimalSurfaceParameter.mFixedBoundary, 0.1f, 10.0f);
		ImGui::InputFloat("Fairness Weight", &mMinimalSurfaceParameter.mLaplacian, 0.1f, 10.0f);
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
	case MyViewerOp::MinimalSurface:
		mMinimalSurfaceOperation.reset(new AAShapeUp::MinimalSurfaceOperation(mGeometrySolverShPtr));
		break;
	default:
		std::cout << "Nothing happened. Not implemented?" << std::endl;
		break;
	}
}

void MyViewer::resetModelToOrigin()
{
	mModel->copyObj(*mModelOrigin);

	mMeshConverterOrigin.setObjModelPtr(mModelOrigin.get());
	mMeshConverterOrigin.generateEigenMatrices();

	mMeshConverterOrigin.updateSourceMesh(AAShapeUp::regenerateNormals(mMeshConverterOrigin.getEigenMesh()), true);

	mMeshConverter.setObjModelPtr(mModel.get());
	mMeshConverter.generateEigenMatrices();

	mMeshConverter.updateSourceMesh(AAShapeUp::regenerateNormals(mMeshConverter.getEigenMesh()), true);
}

void MyViewer::updateReference(ObjModel* objModelPtr)
{
	mMeshConverterReference.setObjModelPtr(objModelPtr);
	mMeshConverterReference.generateEigenMatrices();

	mMeshConverterReference.updateSourceMesh(AAShapeUp::regenerateNormals(mMeshConverterReference.getEigenMesh()), true);
}

