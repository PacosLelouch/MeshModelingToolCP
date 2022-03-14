#pragma once

#define BEGIN_NAMESPACE(Namespace) namespace Namespace {
#define END_NAMESPACE() }

BEGIN_NAMESPACE(AAShapeUp)

using i8 = char;
using ui8 = unsigned char;
using i16 = short;
using ui16 = unsigned short;
using i32 = int;
using ui32 = unsigned int;
using i64 = long long;
using ui64 = unsigned long long;
using f32 = float;
using f64 = double;

using scalar = f32;

constexpr i32 INVALID_INT = -1;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace TimerUtil
{
    using EventID = i32;
}

class NullTimer
{
public:
    using EventID = TimerUtil::EventID;

    virtual EventID recordTime(const char* eventName = nullptr)
    {
        return EventID(-1);
    }

    virtual EventID getTimerEvent(const char* eventName) const
    {
        return EventID(-1);
    }

    virtual double getElapsedTime(EventID startEvent, EventID endEvent) const
    {
        return -1.0;
    }

    virtual void reset()
    {
    }

protected:
};

END_NAMESPACE()
