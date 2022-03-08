#pragma once

#include <memory>
#include "glm/fwd.hpp"

#define DivUp(x, y) ((x) + (y) - 1) / (y)

namespace CUDAExport
{
    enum class MemoryType : unsigned char
    {
        HostPaged,
        HostPinned,
        Device,
    };

    template<typename Type>
    Type* hostPinnedNew(unsigned int count)
    {
        Type* hstPtr = nullptr;
        hostMalloc(reinterpret_cast<void**>(&hstPtr), count * sizeof(Type));
        return hstPtr;
    }

    template<typename Type>
    Type* deviceNew(unsigned int count)
    {
        Type* devPtr = nullptr;
        deviceMalloc(reinterpret_cast<void**>(&devPtr), count * sizeof(Type));
        return devPtr;
    }

    template<typename Type>
    class CUDASharedPtr
    {
        using ElemType = typename Type;
    public:
        CUDASharedPtr() = default;
        CUDASharedPtr(const CUDASharedPtr& c) = default;
        CUDASharedPtr(CUDASharedPtr&& m) = default;
        CUDASharedPtr(const std::shared_ptr<Type>& s, unsigned int inCount = 1, MemoryType inMemoryType = MemoryType::Device)
            : devShPtr(s)
            , count(inCount)
            , memoryType(inMemoryType) {}
        CUDASharedPtr(const Type* p, unsigned int inCount = 1, MemoryType inMemoryType = MemoryType::Device) 
            : devShPtr()
            , count(inCount)
            , memoryType(inMemoryType) 
        {
            switch (inMemoryType)
            {
            case MemoryType::HostPaged:
                devShPtr = std::shared_ptr<Type>(p);
                break;
            case MemoryType::HostPinned:
                devShPtr = std::shared_ptr<Type>(p, hostFree);
                break;
            case MemoryType::Device:
                devShPtr = std::shared_ptr<Type>(p, deviceFree);
                break;
            }
        }
        CUDASharedPtr(unsigned int inCount, MemoryType inMemoryType = MemoryType::Device)
            : devShPtr()
            , count(inCount)
            , memoryType(inMemoryType) 
        {
            Type* p = nullptr;
            switch (inMemoryType)
            {
            case MemoryType::HostPaged:
                p = new Type[inCount];
                devShPtr = std::shared_ptr<Type>(p);
                break;
            case MemoryType::HostPinned:
                p = hostPinnedNew<Type>(inCount);
                devShPtr = std::shared_ptr<Type>(p, hostFree);
                break;
            case MemoryType::Device:
                p = deviceNew<Type>(inCount);
                devShPtr = std::shared_ptr<Type>(p, deviceFree);
                break;
            }
        }


        CUDASharedPtr& operator =(const CUDASharedPtr& c) = default;

        template<class ... Args>
        inline void init(Args&& ... args)
        {
            *this = CUDASharedPtr<Type>(std::forward<Args>(args) ...);
        }

        inline std::shared_ptr<Type>& shPtr()
        {
            return devShPtr;
        }
        inline const std::shared_ptr<Type>& shPtr() const
        {
            return devShPtr;
        }

        inline Type* rawPtr()
        {
            return devShPtr.get();
        }
        inline const Type* rawPtr() const
        {
            return devShPtr.get();
        }

        inline unsigned int getCount() const
        {
            return count;
        }

    protected:
        //inline static void freeDevPtr(Type* p)
        //{
        //    deviceFree(p);
        //}
        std::shared_ptr<Type> devShPtr;
        unsigned int count = 0;
        MemoryType memoryType = MemoryType::Device;
    };

    template<typename Type>
    struct Buffer
    {
        Type* buffer = nullptr;
        unsigned int count = 0;

        GLM_FUNC_DECL Type& operator[](unsigned int i) { return buffer[i]; }
        GLM_FUNC_DECL Type operator[](unsigned int i) const { return buffer[i]; }

        template<typename TypeFrom>
        static Buffer fromCUDASharedPtr(const CUDASharedPtr<TypeFrom>& shPtr)
        {
            static_assert(sizeof(TypeFrom) == sizeof(Type), "sizeof(TypeFrom) != sizeof(Type)");
            return { reinterpret_cast<Type*>(const_cast<TypeFrom*>(shPtr.rawPtr())), shPtr.getCount() };
        }
    };
}