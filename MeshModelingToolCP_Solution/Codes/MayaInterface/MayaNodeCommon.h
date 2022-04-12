#pragma once

#include <maya/MGlobal.h>
#include <maya/MArgList.h>
#include "TypesCommonMinimal.h"

#define MAYA_ATTR_INPUT(attr) \
    (attr).setKeyable(true); \
    (attr).setStorable(true); \
    (attr).setReadable(true); \
    (attr).setWritable(true); 

#define MAYA_ATTR_OUTPUT(attr) \
    (attr).setKeyable(false); \
    (attr).setStorable(false); \
    (attr).setReadable(true); \
    (attr).setWritable(false); 

enum class InputChangedFlag : AAShapeUp::ui8
{
    None = 0,
    Visualization,
    Parameter,
    InputMesh,
    ReferenceMesh,
};


inline InputChangedFlag operator&(const InputChangedFlag f1, const InputChangedFlag f2)
{
    return InputChangedFlag(AAShapeUp::ui8(f1) & AAShapeUp::ui8(f2));
}

inline InputChangedFlag& operator&=(InputChangedFlag& f1, const InputChangedFlag f2)
{
    f1 = (f1 & f2);
    return f1;
}

inline InputChangedFlag operator|(const InputChangedFlag f1, const InputChangedFlag f2)
{
    return InputChangedFlag(AAShapeUp::ui8(f1) | AAShapeUp::ui8(f2));
}

inline InputChangedFlag& operator|=(InputChangedFlag& f1, const InputChangedFlag f2)
{
    f1 = (f1 | f2);
    return f1;
}

namespace std
{
    _NODISCARD inline string to_string(const MString& _Val)
    {
        return _Val.asChar();
    }
}

template<typename Type>
unsigned int parseMayaCommandArg(Type& var, const MArgList& args, const char* shortFlag, const char* longFlag = nullptr, bool display = false)
{
    unsigned int index = args.flagIndex(shortFlag, longFlag);
    if (index != MArgList::kInvalidArgIndex)
    {
        args.get(index + 1, var);
        if (display)
        {
            MGlobal::displayInfo(longFlag + MString(" ") + MString(std::to_string(var).c_str()));
        }
    }
    return index;
}
