// wrapper/hip_wrapper.h
#ifndef HIP_WRAPPER_H
#define HIP_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HIPWrapperHandle HIPWrapperHandle;
typedef struct {
    const char* name;
    const char* arch;
    long long memory_mb;
    double compute_capability;
} CGPUDeviceInfo;

typedef struct {
    CGPUDeviceInfo* devices;
    int count;
} CDeviceInfoArray;

// Create and destroy wrapper
HIPWrapperHandle* CreateHIPWrapper();
void DestroyHIPWrapper(HIPWrapperHandle* handle);

// Initialize the wrapper
int InitializeWrapper(HIPWrapperHandle* handle);

// Process data
int ProcessData(HIPWrapperHandle* handle,
               const unsigned char* input_data, 
               size_t input_size,
               const char* model_type,
               const char** param_keys,
               const char** param_values,
               int param_count,
               unsigned char** output_data,
               size_t* output_size);

// Get device information
CDeviceInfoArray* GetDeviceInfo(HIPWrapperHandle* handle);
void FreeDeviceInfoArray(CDeviceInfoArray* array);

// Free output data
void FreeOutputData(unsigned char* data);

#ifdef __cplusplus
}
#endif

#endif // HIP_WRAPPER_H

// wrapper/hip_wrapper.cpp
#include "hip_wrapper.h"
#include "hip_wrapper.hpp"
#include <cstring>
#include <map>

struct HIPWrapperHandle {
    aiwrapper::HIPWrapper* impl;
};

HIPWrapperHandle* CreateHIPWrapper() {
    try {
        auto handle = new HIPWrapperHandle();
        handle->impl = new aiwrapper::HIPWrapper();
        return handle;
    } catch (const std::exception&) {
        return nullptr;
    }
}

void DestroyHIPWrapper(HIPWrapperHandle* handle) {
    if (handle) {
        delete handle->impl;
        delete handle;
    }
}

int InitializeWrapper(HIPWrapperHandle* handle) {
    if (!handle || !handle->impl) return 0;
    try {
        return handle->impl->Initialize() ? 1 : 0;
    } catch (const std::exception&) {
        return 0;
    }
}

int ProcessData(HIPWrapperHandle* handle,
               const unsigned char* input_data, 
               size_t input_size,
               const char* model_type,
               const char** param_keys,
               const char** param_values,
               int param_count,
               unsigned char** output_data,
               size_t* output_size) {
    if (!handle || !handle->impl) return 0;
    
    try {
        // Convert input data to vector
        std::vector<uint8_t> input(input_data, input_data + input_size);
        
        // Convert parameters to map
        std::map<std::string, std::string> params;
        for (int i = 0; i < param_count; i++) {
            params[param_keys[i]] = param_values[i];
        }
        
        // Process data
        auto result = handle->impl->ProcessData(input, model_type, params);
        
        // Allocate and copy output
        *output_size = result.size();
        *output_data = new unsigned char[*output_size];
        std::memcpy(*output_data, result.data(), *output_size);
        
        return 1;
    } catch (const std::exception&) {
        return 0;
    }
}

CDeviceInfoArray* GetDeviceInfo(HIPWrapperHandle* handle) {
    if (!handle || !handle->impl) return nullptr;
    
    try {
        auto devices = handle->impl->GetDeviceInfo();
        auto array = new CDeviceInfoArray();
        array->count = devices.size();
        array->devices = new CGPUDeviceInfo[array->count];
        
        for (int i = 0; i < array->count; i++) {
            // Allocate and copy strings
            char* name = new char[devices[i].name.length() + 1];
            char* arch = new char[devices[i].arch.length() + 1];
            
            strcpy(name, devices[i].name.c_str());
            strcpy(arch, devices[i].arch.c_str());
            
            array->devices[i].name = name;
            array->devices[i].arch = arch;
            array->devices[i].memory_mb = devices[i].memory_mb;
            array->devices[i].compute_capability = devices[i].compute_capability;
        }
        
        return array;
    } catch (const std::exception&) {
        return nullptr;
    }
}

void FreeDeviceInfoArray(CDeviceInfoArray* array) {
    if (array) {
        for (int i = 0; i < array->count; i++) {
            delete[] array->devices[i].name;
            delete[] array->devices[i].arch;
        }
        delete[] array->devices;
        delete array;
    }
}

void FreeOutputData(unsigned char* data) {
    delete[] data;
}

// Go wrapper (pkg/hip/wrapper.go)
package hip

/*
#cgo CXXFLAGS: -I${SRCDIR}/../../wrapper -std=c++11
#cgo LDFLAGS: -L${SRCDIR}/../../build -lhip_wrapper -lhip
#include <stdlib.h>
#include "hip_wrapper.h"
*/
import "C"
import (
    "errors"
    "runtime"
    "unsafe"
)

type HIPWrapper struct {
    handle *C.HIPWrapperHandle
}

type GPUDevice struct {
    Name              string
    Arch              string
    MemoryMb         int64
    ComputeCapability float64
}

func NewHIPWrapper() (*HIPWrapper, error) {
    handle := C.CreateHIPWrapper()
    if handle == nil {
        return nil, errors.New("failed to create HIP wrapper")
    }
    
    wrapper := &HIPWrapper{handle: handle}
    runtime.SetFinalizer(wrapper, (*HIPWrapper).Destroy)
    
    if C.InitializeWrapper(handle) == 0 {
        wrapper.Destroy()
        return nil, errors.New("failed to initialize HIP wrapper")
    }
    
    return wrapper, nil
}

func (w *HIPWrapper) Destroy() {
    if w.handle != nil {
        C.DestroyHIPWrapper(w.handle)
        w.handle = nil
    }
}

func (w *HIPWrapper) ProcessData(inputData []byte, modelType string, params map[string]string) ([]byte, error) {
    if w.handle == nil {
        return nil, errors.New("wrapper is destroyed")
    }
    
    // Convert model type to C string
    cModelType := C.CString(modelType)
    defer C.free(unsafe.Pointer(cModelType))
    
    // Convert parameters to C arrays
    paramCount := len(params)
    cKeys := make([]*C.char, paramCount)
    cValues := make([]*C.char, paramCount)
    
    i := 0
    for k, v := range params {
        cKeys[i] = C.CString(k)
        cValues[i] = C.CString(v)
        defer C.free(unsafe.Pointer(cKeys[i]))
        defer C.free(unsafe.Pointer(cValues[i]))
        i++
    }
    
    var outputData *C.uchar
    var outputSize C.size_t
    
    result := C.ProcessData(
        w.handle,
        (*C.uchar)(unsafe.Pointer(&inputData[0])),
        C.size_t(len(inputData)),
        cModelType,
        (**C.char)(unsafe.Pointer(&cKeys[0])),
        (**C.char)(unsafe.Pointer(&cValues[0])),
        C.int(paramCount),
        &outputData,
        &outputSize,
    )
    
    if result == 0 {
        return nil, errors.New("processing failed")
    }
    
    // Copy output data to Go slice
    output := make([]byte, outputSize)
    copy(output, unsafe.Slice(outputData, outputSize))
    
    // Free C memory
    C.FreeOutputData(outputData)
    
    return output, nil
}

func (w *HIPWrapper) GetDeviceInfo() ([]GPUDevice, error) {
    if w.handle == nil {
        return nil, errors.New("wrapper is destroyed")
    }
    
    deviceArray := C.GetDeviceInfo(w.handle)
    if deviceArray == nil {
        return nil, errors.New("failed to get device info")
    }
    defer C.FreeDeviceInfoArray(deviceArray)
    
    count := int(deviceArray.count)
    devices := make([]GPUDevice, count)
    
    // Convert C array to Go slice
    cDevices := unsafe.Slice(&deviceArray.devices[0], count)
    
    for i := 0; i < count; i++ {
        devices[i] = GPUDevice{
            Name:              C.GoString(cDevices[i].name),
            Arch:              C.GoString(cDevices[i].arch),
            MemoryMb:         int64(cDevices[i].memory_mb),
            ComputeCapability: float64(cDevices[i].compute_capability),
        }
    }
    
    return devices, nil
}
