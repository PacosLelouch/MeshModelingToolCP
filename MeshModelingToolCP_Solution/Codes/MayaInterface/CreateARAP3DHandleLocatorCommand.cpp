#include "CreateARAP3DHandleLocatorCommand.h"
#include "ARAP3DHandleLocator.h"
#include "ARAP3DNode.h"
#include <maya/MSelectionList.h>
#include <maya/MDagPath.h>
#include <maya/MFnDagNode.h>
#include <maya/MDagModifier.h>
#include <maya/MPointArray.h>
#include <maya/MPlugArray.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MDataHandle.h>

const MString MCreateARAP3DHandleLocatorCommand::commandName = "createARAP3DHandleLocatorInternal";

void* MCreateARAP3DHandleLocatorCommand::creator()
{
    return new MCreateARAP3DHandleLocatorCommand;
}

MStatus MCreateARAP3DHandleLocatorCommand::doIt(const MArgList& args)
{
    MStatus status = MStatus::kSuccess;
    bool displayExecution = false;
    parseMayaCommandArg(displayExecution, args, "-d", "-displayExecution", true);

    char commandBuffer[2048] { 0 };
    char outputBuffer[2048]{ 0 };

    MSelectionList selectionList;
    
    status = MGlobal::getActiveSelectionList(selectionList);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    unsigned int selectionListLength = selectionList.length();
    MDagModifier dagm;
    
    sprintf_s(commandBuffer, "MCreateARAP3DHandleLocatorCommand::doIt(), selected %d objects.", selectionListLength);
    MGlobal::displayInfo(commandBuffer);

    MDagPath meshNodePath;
    MObject meshComponent;
    MFnDagNode meshNodeFn;
    MFnMesh meshFn;
    MStringArray selectionStringArray;
    
    MIntArray vertices;
    MPointArray worldPositions;

    int newLocatorIndex = -1;
    for (unsigned int index = 0; index < selectionListLength; ++index)
    {
        selectionList.getDagPath(index, meshNodePath, meshComponent);

        meshNodeFn.setObject(meshNodePath);
        bool hasMesh = meshNodePath.hasFn(MFn::kMesh);
        MString meshNodeName = meshNodeFn.name();
        MGlobal::displayInfo(meshNodeName + " is selected, " + (hasMesh ? "has mesh." : "doesn't has mesh."));

        MStringArray transformNodeNames;
        sprintf_s(commandBuffer, "listRelatives -p %s;", meshNodeName.asChar());
        status = MGlobal::executeCommand(commandBuffer, transformNodeNames, displayExecution);
        CHECK_MSTATUS_AND_RETURN_IT(status);

        if (transformNodeNames.length() == 0)
        {
            continue;
        }

        MString transformNodeName = transformNodeNames[0];

        // Start get mesh attributes.

        meshFn.setObject(meshNodePath);
        status = meshFn.getPoints(worldPositions, MSpace::kWorld);
        CHECK_MSTATUS_AND_RETURN_IT(status);

        // End get mesh attributes.

        // Start get selection vertices.

        status = selectionList.getSelectionStrings(index, selectionStringArray);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        for (MString& string : selectionStringArray)
        {
            MGlobal::displayInfo("selection:\"" + string + "\"");
        }

        status = findVerticesFromSelections(vertices, selectionStringArray);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        for (auto vertex : vertices)
        {
            sprintf_s(outputBuffer, "vertex: %d, world pos <%.3f, %.3f, %.3f>", vertex, worldPositions[vertex].x, worldPositions[vertex].y, worldPositions[vertex].z);
            MGlobal::displayInfo(outputBuffer);
        }

        // End get selection vertices.

        // Start get deformer node.

        MStringArray deformerNodeNames;
        status = findDeformerNodeNamesFromSelectedShape(deformerNodeNames, meshNodeName, MARAP3DNode::nodeName, false);
        for (MString& string : deformerNodeNames)
        {
            MGlobal::displayInfo("deformer:\"" + string + "\"");
        }

        if (!hasMesh || deformerNodeNames.length() == 0)
        {
            continue;
        }

        MString deformerNodeName = deformerNodeNames[0];
        status = MGlobal::selectByName(deformerNodeName, MGlobal::kReplaceList);
        CHECK_MSTATUS_AND_RETURN_IT(status);

        MSelectionList deformerNodeList;
        status = MGlobal::getActiveSelectionList(deformerNodeList);
        CHECK_MSTATUS_AND_RETURN_IT(status);

        //MStringArray deformerNodeSelectedNames;
        //deformerNodeList.getSelectionStrings(deformerNodeSelectedNames);
        //sprintf_s(outputBuffer, "Get %d deformer node(s).", deformerNodeList.length());
        //MGlobal::displayInfo(outputBuffer);
        //for (MString& string : deformerNodeSelectedNames)
        //{
        //    sprintf_s(outputBuffer, "%s selected.", string.asChar());
        //    MGlobal::displayInfo(outputBuffer);
        //}

        //status = MGlobal::setActiveSelectionList(selectionList);
        //CHECK_MSTATUS_AND_RETURN_IT(status);

        //MDagPath deformerNodePath;
        //MObject deformerComponent;
        MObject deformerNode;
        MFnDagNode deformerNodeFn;

        deformerNodeList.getDependNode(0, deformerNode);
        //deformerNodeList.getDagPath(0, deformerNodePath, deformerComponent);
        //deformerNodeFn.setObject(deformerNodePath);
        deformerNodeFn.setObject(deformerNode);

        // End get deformer node.

        // Start get attributes from deformer.

        //MGlobal::displayInfo("Deformer node path is valid: " + MString(deformerNodePath.isValid() ? "true" : "false"));
        //MGlobal::displayInfo("Deformer node path: " + deformerNodePath.fullPathName());
        //MGlobal::displayInfo("Deformer node apiTypeStr: " + MString(deformerNode.apiTypeStr()));

        MPlug plugPositions(deformerNode, MARAP3DNode::aHandlePositions); 
        MPlug plugIndices(deformerNode, MARAP3DNode::aHandleIndices);
        
        //MGlobal::displayInfo("plug: " + plugPositions.name());
        //MGlobal::displayInfo("plug: " + plugIndices.name());

        if (!plugPositions.isArray())
        {
            sprintf_s(outputBuffer, "plugPositions not array, error...");
            MGlobal::displayError(outputBuffer);
            return MStatus::kFailure;
        }

        if (!plugIndices.isArray())
        {
            sprintf_s(outputBuffer, "plugIndices not array, error...");
            MGlobal::displayError(outputBuffer);
            return MStatus::kFailure;
        }

        unsigned int numHandlePositions = plugPositions.evaluateNumElements(&status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        unsigned int numHandleIndices = plugIndices.evaluateNumElements(&status);
        CHECK_MSTATUS_AND_RETURN_IT(status);

        if (numHandlePositions != numHandleIndices)
        {
            sprintf_s(outputBuffer, "%d != %d, error...", numHandlePositions, numHandleIndices);
            MGlobal::displayError(outputBuffer);
            return MStatus::kFailure;
        }

        std::unordered_set<unsigned int> logicalIndices;
        for (unsigned int n = 0; n < numHandleIndices; ++n)
        {
            logicalIndices.insert(plugIndices[n].logicalIndex(&status));
        }

        //newLocatorIndex = numHandlePositions == 0 ? 0 : plugPositions.elementByPhysicalIndex(numHandlePositions - 1, &status).logicalIndex() + 1;
        newLocatorIndex = 0;
        while (logicalIndices.find(newLocatorIndex) != logicalIndices.end())
        {
            ++newLocatorIndex;
        }

        for (unsigned int vi = 0; vi < vertices.length(); ++vi)
        {
            unsigned int vidx = vertices[vi];
            //sprintf_s(commandBuffer, "createNode %s -parent %s;", MARAP3DHandleLocatorNode::nodeName.asChar(), transformNodeName.asChar());
            sprintf_s(commandBuffer, "createNode %s;", MARAP3DHandleLocatorNode::nodeName.asChar());
            status = MGlobal::executeCommand(commandBuffer, displayExecution);
            CHECK_MSTATUS_AND_RETURN_IT(status);
            
            MStringArray newLocatorNames;
            status = MGlobal::executeCommand("ls -selection;", newLocatorNames, displayExecution);
            if (newLocatorNames.length() == 0)
            {
                sprintf_s(outputBuffer, "fail to create new locator, error...");
                MGlobal::displayError(outputBuffer);
                return MStatus::kFailure;
            }
            MString newLocatorName = newLocatorNames[0];

            MStringArray locatorTransformNodeNames;
            sprintf_s(commandBuffer, "listRelatives -p %s;", newLocatorName.asChar());
            status = MGlobal::executeCommand(commandBuffer, locatorTransformNodeNames, displayExecution);
            CHECK_MSTATUS_AND_RETURN_IT(status);

            if (locatorTransformNodeNames.length() == 0)
            {
                continue;
            }

            MString locatorTransformNodeName = locatorTransformNodeNames[0];

            sprintf_s(commandBuffer, "setAttr %s.vertexIndex %d;", newLocatorName.asChar(), vidx);
            status = MGlobal::executeCommand(commandBuffer, displayExecution);
            CHECK_MSTATUS_AND_RETURN_IT(status);

            sprintf_s(commandBuffer, "setAttr %s.translate %f %f %f;", locatorTransformNodeName.asChar(), worldPositions[vidx].x, worldPositions[vidx].y, worldPositions[vidx].z);
            status = MGlobal::executeCommand(commandBuffer, displayExecution);
            CHECK_MSTATUS_AND_RETURN_IT(status);

            // Start make connections.

            MStatus status = plugPositions.selectAncestorLogicalIndex(newLocatorIndex, MARAP3DNode::aHandlePositions);
            //MDataHandle dataHandle;
            //dataHandle.set3Double(worldPositions[vidx].x, worldPositions[vidx].y, worldPositions[vidx].z);
            //plugPositions.setValue(dataHandle);

            sprintf_s(commandBuffer, "connectAttr %s.worldPosition[0] %s.handlePositions[%d];", newLocatorName.asChar(), deformerNodeName.asChar(), newLocatorIndex);
            status = MGlobal::executeCommand(commandBuffer, displayExecution);
            CHECK_MSTATUS_AND_RETURN_IT(status);

            sprintf_s(commandBuffer, "connectAttr %s.vertexIndex %s.handleIndices[%d];", newLocatorName.asChar(), deformerNodeName.asChar(), newLocatorIndex);
            status = MGlobal::executeCommand(commandBuffer, displayExecution);
            CHECK_MSTATUS_AND_RETURN_IT(status);

            //++newLocatorIndex;
            logicalIndices.insert(newLocatorIndex);
            while (logicalIndices.find(newLocatorIndex) != logicalIndices.end())
            {
                ++newLocatorIndex;
            }

            // End make connections.
        }

        // End get attributes from deformer.

        // Only the first object.
        break;
    }

    return status;
}

MStatus MCreateARAP3DHandleLocatorCommand::findDeformerNodeNamesFromSelectedShape(MStringArray& deformerNodeNames, const MString& shapeName, const MString& deformerType, bool displayExecution)
{
    MStatus status = deformerNodeNames.clear();
    CHECK_MSTATUS_AND_RETURN_IT(status);

    char buffer[2048]{ 0 };

    MStringArray connectedObjectSetsResultArray;
    sprintf_s(buffer, "listConnections -type objectSet %s", shapeName.asChar());

    status = MGlobal::executeCommand(buffer, connectedObjectSetsResultArray, displayExecution);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    CHECK_MSTATUS_AND_RETURN_IT(status);

    MStringArray connectedDeformersResultArray;
    for (const MString& connectedObjectSet : connectedObjectSetsResultArray)
    {
        sprintf_s(buffer, "listConnections -type %s %s", deformerType.asChar(), connectedObjectSet.asChar());
        status = MGlobal::executeCommand(buffer, connectedDeformersResultArray, displayExecution);
        CHECK_MSTATUS_AND_RETURN_IT(status);

        if (connectedDeformersResultArray.length() > 0)
        {
            deformerNodeNames = connectedDeformersResultArray;
            return status;
        }
    }

    return status;
}

MStatus MCreateARAP3DHandleLocatorCommand::findVerticesFromSelections(MIntArray& vertices, const MStringArray& selections)
{
    MStatus status = vertices.clear();
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MStringArray token0, token1, token2;
    for (unsigned int i = 0; i < selections.length(); ++i)
    {
        token0.clear();
        token1.clear();
        token2.clear();
        const MString& selection = selections[i];
        status = selection.split('[', token0);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        if (token0.length() < 2)
        {
            continue;
        }
        MString beforeParenthesis = token0[token0.length() - 2];
        MString vtxToken = beforeParenthesis.substring(beforeParenthesis.length() - 3, beforeParenthesis.length());
        if (vtxToken != "vtx")
        {
            continue;
        }
        status = token0[token0.length() - 1].split(']', token1);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        if (token1.length() == 0)
        {
            continue;
        }

        status = token1[0].split(':', token2);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        if (token2.length() == 2)
        {
            int start = token2[0].asInt();
            int end = token2[1].asInt();
            for (int num = start; num <= end; ++num)
            {
                vertices.append(num);
            }
        }
        else
        {
            vertices.append(token1[0].asInt());
        }
    }

    return status;
}
