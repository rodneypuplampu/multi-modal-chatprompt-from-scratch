// proto/ai_service.proto
syntax = "proto3";

package aiservice;
option go_package = "github.com/yourusername/aibridge/proto";

service AIService {
  // Send input data for processing
  rpc ProcessData (ProcessRequest) returns (ProcessResponse) {}
  
  // Get GPU device information
  rpc GetDeviceInfo (DeviceInfoRequest) returns (DeviceInfoResponse) {}
  
  // Stream results for long-running operations
  rpc StreamResults (ProcessRequest) returns (stream ProcessResponse) {}
}

message ProcessRequest {
  bytes input_data = 1;
  string model_type = 2;
  map<string, string> parameters = 3;
}

message ProcessResponse {
  bytes output_data = 1;
  float computation_time = 2;
  string error_message = 3;
}

message DeviceInfoRequest {}

message DeviceInfoResponse {
  repeated GPUDevice devices = 1;
}

message GPUDevice {
  string name = 1;
  string arch = 2;
  int64 memory_mb = 3;
  double compute_capability = 4;
}

// HIP C++ Header (include/hip_wrapper.hpp)
#pragma once
#include <hip/hip_runtime.h>
#include <vector>
#include <string>

namespace aiwrapper {

struct GPUDeviceInfo {
    std::string name;
    std::string arch;
    int64_t memory_mb;
    double compute_capability;
};

class HIPWrapper {
public:
    HIPWrapper();
    ~HIPWrapper();

    // Initialize HIP and load model
    bool Initialize();
    
    // Process data using HIP
    std::vector<uint8_t> ProcessData(const std::vector<uint8_t>& input_data,
                                   const std::string& model_type,
                                   const std::map<std::string, std::string>& params);
    
    // Get device information
    std::vector<GPUDeviceInfo> GetDeviceInfo();

private:
    void* model_handle_;
    hipStream_t stream_;
};

} // namespace aiwrapper

// HIP C++ Implementation (src/hip_wrapper.cpp)
#include "hip_wrapper.hpp"
#include <stdexcept>

namespace aiwrapper {

HIPWrapper::HIPWrapper() : model_handle_(nullptr) {
    if (hipInit(0) != hipSuccess) {
        throw std::runtime_error("Failed to initialize HIP runtime");
    }
    hipStreamCreate(&stream_);
}

HIPWrapper::~HIPWrapper() {
    if (stream_) {
        hipStreamDestroy(stream_);
    }
}

bool HIPWrapper::Initialize() {
    // Initialize your AI model here
    return true;
}

std::vector<uint8_t> HIPWrapper::ProcessData(
    const std::vector<uint8_t>& input_data,
    const std::string& model_type,
    const std::map<std::string, std::string>& params) {
    
    // Allocate GPU memory
    void* d_input;
    void* d_output;
    size_t input_size = input_data.size();
    size_t output_size = input_size; // Adjust based on your needs
    
    hipMalloc(&d_input, input_size);
    hipMalloc(&d_output, output_size);
    
    // Copy input to GPU
    hipMemcpyAsync(d_input, input_data.data(), input_size, 
                   hipMemcpyHostToDevice, stream_);
    
    // Process data (implement your kernel launch here)
    // hipLaunchKernelGGL(...);
    
    // Copy result back
    std::vector<uint8_t> output(output_size);
    hipMemcpyAsync(output.data(), d_output, output_size,
                   hipMemcpyDeviceToHost, stream_);
    
    // Synchronize and clean up
    hipStreamSynchronize(stream_);
    hipFree(d_input);
    hipFree(d_output);
    
    return output;
}

std::vector<GPUDeviceInfo> HIPWrapper::GetDeviceInfo() {
    std::vector<GPUDeviceInfo> devices;
    int device_count;
    hipGetDeviceCount(&device_count);
    
    for (int i = 0; i < device_count; i++) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);
        
        GPUDeviceInfo info;
        info.name = props.name;
        info.arch = std::to_string(props.major) + "." + 
                   std::to_string(props.minor);
        info.memory_mb = props.totalGlobalMem / (1024 * 1024);
        info.compute_capability = props.major + props.minor / 10.0;
        
        devices.push_back(info);
    }
    
    return devices;
}

} // namespace aiwrapper

// Go Server Implementation (cmd/server/main.go)
package main

import (
    "context"
    "log"
    "net"
    
    pb "github.com/yourusername/aibridge/proto"
    "google.golang.org/grpc"
)

type server struct {
    pb.UnimplementedAIServiceServer
    wrapper *HIPWrapper
}

func (s *server) ProcessData(ctx context.Context, req *pb.ProcessRequest) (*pb.ProcessResponse, error) {
    // Convert parameters to C++ map
    params := make(map[string]string)
    for k, v := range req.Parameters {
        params[k] = v
    }
    
    // Process using HIP wrapper
    output, err := s.wrapper.ProcessData(req.InputData, req.ModelType, params)
    if err != nil {
        return &pb.ProcessResponse{
            ErrorMessage: err.Error(),
        }, nil
    }
    
    return &pb.ProcessResponse{
        OutputData: output,
    }, nil
}

func (s *server) GetDeviceInfo(ctx context.Context, req *pb.DeviceInfoRequest) (*pb.DeviceInfoResponse, error) {
    devices := s.wrapper.GetDeviceInfo()
    
    pbDevices := make([]*pb.GPUDevice, len(devices))
    for i, d := range devices {
        pbDevices[i] = &pb.GPUDevice{
            Name:              d.Name,
            Arch:              d.Arch,
            MemoryMb:         d.MemoryMb,
            ComputeCapability: d.ComputeCapability,
        }
    }
    
    return &pb.DeviceInfoResponse{
        Devices: pbDevices,
    }, nil
}

func main() {
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("failed to listen: %v", err)
    }
    
    // Initialize HIP wrapper
    wrapper, err := NewHIPWrapper()
    if err != nil {
        log.Fatalf("failed to create HIP wrapper: %v", err)
    }
    
    s := grpc.NewServer()
    pb.RegisterAIServiceServer(s, &server{wrapper: wrapper})
    
    log.Printf("server listening at %v", lis.Addr())
    if err := s.Serve(lis); err != nil {
        log.Fatalf("failed to serve: %v", err)
    }
}

// Go Client Example (cmd/client/main.go)
package main

import (
    "context"
    "log"
    "time"
    
    pb "github.com/yourusername/aibridge/proto"
    "google.golang.org/grpc"
)

func main() {
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatalf("did not connect: %v", err)
    }
    defer conn.Close()
    
    c := pb.NewAIServiceClient(conn)
    ctx, cancel := context.WithTimeout(context.Background(), time.Second)
    defer cancel()
    
    // Get device info
    deviceInfo, err := c.GetDeviceInfo(ctx, &pb.DeviceInfoRequest{})
    if err != nil {
        log.Fatalf("could not get device info: %v", err)
    }
    log.Printf("Device info: %v", deviceInfo)
    
    // Process some data
    resp, err := c.ProcessData(ctx, &pb.ProcessRequest{
        InputData:  []byte("some input data"),
        ModelType:  "example_model",
        Parameters: map[string]string{"param1": "value1"},
    })
    if err != nil {
        log.Fatalf("could not process data: %v", err)
    }
    log.Printf("Response: %v", resp)
}
