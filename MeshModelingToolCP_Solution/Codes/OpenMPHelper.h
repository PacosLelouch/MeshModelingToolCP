#pragma once

#include "TypesCommonMinimal.h"

#ifdef USE_OPENMP
#include <omp.h>
#ifdef USE_MSVC
#define OMP_PARALLEL __pragma(omp parallel)
#define OMP_FOR __pragma(omp for)
#define OMP_SINGLE __pragma(omp single)
#define OMP_SECTIONS __pragma(omp sections)
#define OMP_SECTION __pragma(omp section)
#else
#define OMP_PARALLEL _Pragma("omp parallel")
#define OMP_FOR _Pragma("omp for")
#define OMP_SINGLE _Pragma("omp single")
#define OMP_SECTIONS _Pragma("omp sections")
#define OMP_SECTION _Pragma("omp section")
#endif
#else
#include <ctime>
#define OMP_PARALLEL
#define OMP_FOR
#define OMP_SINGLE
#define OMP_SECTIONS
#define OMP_SECTION
#endif

#include <cassert>
#include <vector>
#include <unordered_map>
#include <string>

BEGIN_NAMESPACE(AAShapeUp)

class OpenMPTimer : public NullTimer
{
public:
    using NullTimer::EventID;

    virtual EventID recordTime(const char* eventName = nullptr) override
    {
        if (eventName != nullptr)
        {
            auto it = m_nameToIdxMap.find(eventName);
            if (it == m_nameToIdxMap.end())
            {
                m_names.push_back(eventName);
                m_nameToIdxMap.insert(std::make_pair(std::string(eventName), m_timeValues.size()));
            }
            else
            {
                return EventID(it->second);
            }
        }

        EventID id = EventID(m_timeValues.size());
        
#ifdef USE_OPENMP
        m_timeValues.push_back(omp_get_wtime());
#else // !USE_OPENMP
        m_timeValues.push_back(clock());
#endif // USE_OPENMP

        return id;
    }

    virtual EventID getTimerEvent(const char* eventName) const override
    {
        auto it = m_nameToIdxMap.find(eventName);
        if (it == m_nameToIdxMap.end())
        {
            return EventID(-1);
        }
        return EventID(it->second);
    }

    virtual double getElapsedTime(EventID startEvent, EventID endEvent) const override
    {
        assert(startEvent >= 0 && startEvent < static_cast<EventID>(m_timeValues.size()));
        assert(endEvent >= 0 && endEvent < static_cast<EventID>(m_timeValues.size()));
        assert(startEvent < endEvent);

#ifdef USE_OPENMP
        return m_timeValues[endEvent] - m_timeValues[startEvent];
#else // !USE_OPENMP
        return double(m_timeValues[endEvent] - m_timeValues[startEvent]) / double(CLOCKS_PER_SEC);
#endif // USE_OPENMP
    }

    virtual void reset() override
    {
        m_timeValues.clear();
    }

protected:
#ifdef USE_OPENMP
    std::vector<double> m_timeValues;
#else // !USE_OPENMP
    std::vector<clock_t> m_timeValues;
#endif // USE_OPENMP
    std::vector<std::string> m_names;
    std::unordered_map<std::string, size_t> m_nameToIdxMap;
};

END_NAMESPACE()
