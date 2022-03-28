#include "pch.h"
#include "PlanarizationNode.h"
#include "maya/MItGeometry.h"

const MTypeId MPlanarizationNode::id = 0x01000001;
const MString MPlanarizationNode::nodeName = "planarizationNode";

void* MPlanarizationNode::creator()
{
    return new MPlanarizationNode;
}

MStatus MPlanarizationNode::initialize()
{
    MStatus status = Super::initialize();
    CHECK_MSTATUS_AND_RETURN_IT(status);

    //TODO
    return MStatus::kSuccess;
}

MStatus MPlanarizationNode::deform(MDataBlock& block, MItGeometry& iter, const MMatrix& mat, unsigned int multiIndex)
{
    //TODO
    return MStatus::kSuccess;
}
