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
