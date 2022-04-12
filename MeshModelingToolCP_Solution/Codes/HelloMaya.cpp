#include "MayaExport.h"

#if defined _WIN32 || defined _WIN64
    #include <Windows.h>
#else
    #include <stdio.h>
#endif

#include "CUDATest.h"
#include "CUDAExport.h"

//MAYA_EXPORT void ExampleFunction()
//{
//    CUDATest::CUDATest::CUDATestFunction();
////#if defined _WIN32 || defined _WIN64
////	MessageBox(NULL, TEXT("Loaded ExampleLibrary.dll from Third Party Plugin sample 111."), TEXT("Third Party Plugin"), MB_OK);
////#else
////    printf("Loaded ExampleLibrary from Third Party Plugin sample");
////#endif
//}
#include "HelloMaya.h"
#include "MayaInterface/PlanarizationNode.h"
#include "MayaInterface/ARAP3DNode.h"
#include "MayaInterface/TestBoundingSphereNode.h"
#include "MayaInterface/ARAP3DHandleLocator.h"
#include "MayaInterface/CreateARAP3DHandleLocatorCommand.h"
#include <maya/MFnPlugin.h>
#include <string>
// Viewport 2.0 includes
#include <maya/MDrawRegistry.h>
//#include <maya/MPxDrawOverride.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MHWGeometryUtilities.h>
#include <maya/MPointArray.h>
#include <maya/MGlobal.h>
#include <maya/MEventMessage.h>
#include <maya/MFnDependencyNode.h>

#if _COMPUTE_USING_CUDA

#endif // _COMPUTE_USING_CUDA

const MString AAShapeUp_Script_MenuCommand_Name = "AAShapeUp_MenuCommand";

const MString AAShapeUp_Menu_Name = "AAShapeUp";
const MString AAShapeUp_Menu_Label = "AA Shape-Up";

const MString AAShapeUp_MenuItem_Planarization_Label = "Planarization";
const MString AAShapeUp_MenuItem_Planarization_Command = "AAShapeUp_createPlanarizationNode";
const MString AAShapeUp_MenuItem_Planarization_Name = "planarizationMenuItem";

const MString AAShapeUp_MenuItem_ARAP3D_Label = "As-Rigid-As-Possible Deformation";
const MString AAShapeUp_MenuItem_ARAP3D_Command = "AAShapeUp_createARAP3DNode";
const MString AAShapeUp_MenuItem_ARAP3D_Name = "ARAP3DMenuItem";

const MString AAShapeUp_MenuItem_TestBoundingSphere_Label = "Test Bounding Sphere";
const MString AAShapeUp_MenuItem_TestBoundingSphere_Command = "AAShapeUp_createTestBoundingSphereNode";
const MString AAShapeUp_MenuItem_TestBoundingSphere_Name = "testBoundingSphereMenuItem";

const MString AAShapeUp_MenuItem_ARAP3DHandleLocator_Label = "Deformation Handle Locator";
const MString AAShapeUp_MenuItem_ARAP3DHandleLocator_Command = "AAShapeUp_createARAP3DHandleLocator";
const MString AAShapeUp_MenuItem_ARAP3DHandleLocator_Name = "ARAP3DHandleLocatorMenuItem";

static MStatus createMenu(const MString& pluginPath)
{
    MStatus status = MStatus::kSuccess;
    char commandBuffer[4096]{ 0 };
    sprintf_s(commandBuffer, 
        R"(source "%s/../scripts/%s.mel";)", 
        pluginPath.asChar(), AAShapeUp_Script_MenuCommand_Name.asChar());
    status = MGlobal::executeCommand(commandBuffer);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MGlobal::displayInfo(commandBuffer);

    sprintf_s(commandBuffer,
        R"(menu -label "%s" -tearOff true -parent MayaWindow "%s";
menuItem -divider true -dividerLabel "Operation";
menuItem -label "%s" -command "%s" "%s";
menuItem -label "%s" -command "%s" "%s";
menuItem -label "%s" -command "%s" "%s";
menuItem -divider true -dividerLabel "Additional";
menuItem -label "%s" -command "%s" "%s";)",
AAShapeUp_Menu_Label.asChar(), AAShapeUp_Menu_Name.asChar(),
AAShapeUp_MenuItem_Planarization_Label.asChar(), AAShapeUp_MenuItem_Planarization_Command.asChar(), AAShapeUp_MenuItem_Planarization_Name.asChar(),
AAShapeUp_MenuItem_ARAP3D_Label.asChar(), AAShapeUp_MenuItem_ARAP3D_Command.asChar(), AAShapeUp_MenuItem_ARAP3D_Name.asChar(), 
AAShapeUp_MenuItem_TestBoundingSphere_Label.asChar(), AAShapeUp_MenuItem_TestBoundingSphere_Command.asChar(), AAShapeUp_MenuItem_TestBoundingSphere_Name.asChar(), 
AAShapeUp_MenuItem_ARAP3DHandleLocator_Label.asChar(), AAShapeUp_MenuItem_ARAP3DHandleLocator_Command.asChar(), AAShapeUp_MenuItem_ARAP3DHandleLocator_Name.asChar());
    status = MGlobal::executeCommand(commandBuffer);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MGlobal::displayInfo("AA Shape-Up Menu created.");
    return status;
}

static MStatus deleteMenu(const MString& pluginPath)
{
    MStatus status = MStatus::kSuccess;
    char commandBuffer[1024]{ 0 };

    sprintf_s(commandBuffer, R"(deleteUI "%s";)", 
        AAShapeUp_Menu_Name.asChar());
    status = MGlobal::executeCommand(commandBuffer);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MGlobal::displayInfo("AA Shape-Up Menu deleted.");
    return status;
}

// Maya Plugin creator function
void *HelloMaya::creator()
{
    return new HelloMaya;
}
// Plugin doIt function
MStatus HelloMaya::doIt(const MArgList& argList)
{
    MStatus status;
    MGlobal::displayInfo("Hello World!");
    // <<<your code goes here>>>
    std::string cudaStatusStr;
    CUDATest::CUDATest::CUDATestFunction(true, &cudaStatusStr);
    MString cudaStatusStrM = cudaStatusStr.c_str();
    MGlobal::displayInfo(cudaStatusStrM);

    size_t start_pos = 0;
    while((start_pos = cudaStatusStr.find("\n", start_pos)) != std::string::npos) {
        cudaStatusStr.replace(start_pos, sizeof("\n") - 1, "\\n");
        start_pos += sizeof("\\n") - 1;
    }

    MString command = "confirmDialog -title \"Hello Maya\" -message \"^1s\" -button \"OK\" -defaultButton \"OK\" -messageAlign \"center\"";
    command.format(command, MString(cudaStatusStr.c_str()));
    MGlobal::executeCommand(command);

    return status;
}
// Initialize Maya Plugin upon loading
MLL_EXPORT MStatus initializePlugin(MObject obj)
{
    MStatus status = MStatus::kSuccess;
    MFnPlugin plugin( obj, "PacosLelouch & cyy0915", "1.0", "Any");
    status = plugin.registerCommand("helloMaya", HelloMaya::creator );
    if (!status)
    {
        status.perror( "registerCommand helloMaya failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    status = plugin.registerCommand(MCreateARAP3DHandleLocatorCommand::commandName, MCreateARAP3DHandleLocatorCommand::creator);
    if (!status)
    {
        status.perror( "registerCommand " + MCreateARAP3DHandleLocatorCommand::commandName + " failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    
    status = plugin.registerNode(
        MPlanarizationNode::nodeName, MPlanarizationNode::id,
        MPlanarizationNode::creator, MPlanarizationNode::initialize, MPxNode::kDeformerNode);
    if (!status)
    {
        status.perror( "registerNode " + MPlanarizationNode::nodeName + " failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    status = plugin.registerNode(
        MARAP3DNode::nodeName, MARAP3DNode::id,
        MARAP3DNode::creator, MARAP3DNode::initialize, MPxNode::kDeformerNode);
    if (!status)
    {
        status.perror( "registerNode " + MARAP3DNode::nodeName + " failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    status = plugin.registerNode(
        MTestBoundingSphereNode::nodeName, MTestBoundingSphereNode::id,
        MTestBoundingSphereNode::creator, MTestBoundingSphereNode::initialize, MPxNode::kDeformerNode);
    if (!status)
    {
        status.perror( "registerNode " + MTestBoundingSphereNode::nodeName + " failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    status = plugin.registerNode(
        MARAP3DHandleLocatorNode::nodeName, MARAP3DHandleLocatorNode::id,
        MARAP3DHandleLocatorNode::creator, MARAP3DHandleLocatorNode::initialize, MPxNode::kLocatorNode, 
        &MARAP3DHandleLocatorNode::drawDbClassification);
    if (!status)
    {
        status.perror( "registerNode " + MARAP3DHandleLocatorNode::nodeName + " failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    //status = MHWRender::MDrawRegistry::registerDrawOverrideCreator(
    //    MARAP3DHandleLocatorNode::drawDbClassification,
    //    MARAP3DHandleLocatorNode::drawRegistrantId,
    //    MARAP3DHandleLocatorDrawOverride::creator);
    //if (!status) {
    //    status.perror("registerDrawOverrideCreator failed");
    //    CHECK_MSTATUS_AND_RETURN_IT(status);
    //}
    status = MHWRender::MDrawRegistry::registerGeometryOverrideCreator(
        MARAP3DHandleLocatorNode::drawDbClassification,
        MARAP3DHandleLocatorNode::drawRegistrantId,
        MARAP3DHandleLocatorGeometryOverride::creator);
    if (!status) {
        status.perror("registerGeometryOverrideCreator failed");
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    MString path = plugin.loadPath(&status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = createMenu(path);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    return status;
}
// Cleanup Plugin upon unloading
MLL_EXPORT MStatus uninitializePlugin(MObject obj)
{
    MStatus status = MStatus::kSuccess;
    MFnPlugin plugin(obj);
    //MGeometryOptimizerNode::initialized = false;

    MString path = plugin.loadPath(&status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = deleteMenu(path);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    //status = MHWRender::MDrawRegistry::deregisterDrawOverrideCreator(
    //    MARAP3DHandleLocatorNode::drawDbClassification,
    //    MARAP3DHandleLocatorNode::drawRegistrantId);
    //if (!status) {
    //    status.perror("deregisterDrawOverrideCreator failed");
    //    CHECK_MSTATUS_AND_RETURN_IT(status);
    //}
    status = MHWRender::MDrawRegistry::deregisterGeometryOverrideCreator(
        MARAP3DHandleLocatorNode::drawDbClassification,
        MARAP3DHandleLocatorNode::drawRegistrantId);
    if (!status) {
        status.perror("deregisterGeometryOverrideCreator failed");
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    status = plugin.deregisterNode(MARAP3DHandleLocatorNode::id);
    if(!status)
    {
        status.perror( "deregisterNode " + MARAP3DHandleLocatorNode::nodeName + " failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    status = plugin.deregisterNode(MTestBoundingSphereNode::id);
    if(!status)
    {
        status.perror( "deregisterNode " + MTestBoundingSphereNode::nodeName + " failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    status = plugin.deregisterNode(MARAP3DNode::id);
    if(!status)
    {
        status.perror( "deregisterNode " + MARAP3DNode::nodeName + " failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    status = plugin.deregisterNode(MPlanarizationNode::id);
    if(!status)
    {
        status.perror( "deregisterNode " + MPlanarizationNode::nodeName + " failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    status = plugin.deregisterCommand("helloMaya");
    if(!status)
    {
        status.perror( "deregisterCommand helloMaya failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    status = plugin.deregisterCommand(MCreateARAP3DHandleLocatorCommand::commandName);
    if (!status)
    {
        status.perror( "deregisterCommand " + MCreateARAP3DHandleLocatorCommand::commandName + " failed" );
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    return status;
}

#include "Dummy.h"

void DummyTestLink()
{
    int* test = nullptr;
    CUDAExport::hostMalloc(reinterpret_cast<void**>(&test), 1);
    CUDAExport::hostFree(test);
    AAShapeUp::DummyTestCompilation();
}