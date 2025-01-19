Create the CGO bindings to connect the Go code with the HIP C++ wrapper and comprehensive 
CGO bindings for the HIP wrapper. Here's a breakdown of the key components:

1. **C Interface Header (`hip_wrapper.h`)**:
   - Pure C interface for the C++ wrapper
   - Opaque handle type for the wrapper instance
   - C-compatible structs for device information
   - Memory management functions

2. **C++ Implementation (`hip_wrapper.cpp`)**:
   - Implements the C interface
   - Handles conversion between C++ and C types
   - Manages memory allocation/deallocation
   - Error handling with try/catch blocks

3. **Go Wrapper (`wrapper.go`)**:
   - High-level Go interface
   - Automatic memory management with finalizers
   - Type conversions between Go and C
   - Error handling and safety checks

To use these bindings:

1. Build the C++ library:
```bash
# In the project root
mkdir build && cd build
cmake ..
make
```

2. Use in your Go code:
```go
package main

import (
    "log"
    "github.com/yourusername/aibridge/pkg/hip"
)

func main() {
    // Create wrapper
    wrapper, err := hip.NewHIPWrapper()
    if err != nil {
        log.Fatal(err)
    }
    defer wrapper.Destroy()
    
    // Get device info
    devices, err := wrapper.GetDeviceInfo()
    if err != nil {
        log.Fatal(err)
    }
    
    // Process some data
    input := []byte("some data")
    params := map[string]string{
        "param1": "value1",
    }
    
    output, err := wrapper.ProcessData(input, "model_type", params)
    if err != nil {
        log.Fatal(err)
    }
    
    // Use output...
}
```

Key features of these bindings:

1. **Memory Safety**:
   - Proper cleanup of C++ objects
   - Automatic memory management in Go
   - Prevention of memory leaks

2. **Type Safety**:
   - Safe conversions between Go and C types
   - Error checking for null pointers
   - Bounds checking for arrays

3. **Error Handling**:
   - C++ exceptions converted to Go errors
   - Detailed error messages
   - Safe cleanup on errors

Would you like me to:
1. Add more error details/logging?
2. Add support for asynchronous operations?
3. Implement specific GPU kernel integration?
4. Add configuration options?
