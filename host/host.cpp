/* host/host.cpp - MODIFIED */
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <map>
#include <numeric>

// Open Enclave API
#include <openenclave/host.h>
#include <openenclave/bits/result.h>

// ONNX Runtime C API
#include <onnxruntime_c_api.h>

#include "enclave_u.h"

// --- Helper Functions and Macros (keep as is) ---
#define OE_HOST_CHECK(oe_result, function_name) // ... keep original macro

// --- Host-side ONNX Runtime Globals (keep as is) ---
static OrtEnv* g_host_ort_env = nullptr;
static const OrtApi* g_ort_api = nullptr;
static std::map<uint64_t, OrtSession*> g_host_onnx_sessions;
static uint64_t g_next_host_session_handle = 1;

// --- File Loading (keep as is) ---
std::vector<unsigned char> load_file_to_buffer(const std::string& filepath); // ... keep original function

// NEW: Helper to release ONNX C-style strings
struct OrtStringReleaser {
    OrtAllocator* allocator;
    char* str;
    OrtStringReleaser(OrtAllocator* alloc, char* s) : allocator(alloc), str(s) {}
    ~OrtStringReleaser() {
        if (str && allocator) {
            allocator->Free(allocator, str);
        }
    }
};


// --- OCALL Implementations (Modified for generic models) ---
oe_result_t ocall_onnx_load_model(
    uint64_t* host_session_handle_out,
    const unsigned char* model_data,
    size_t model_data_len) {

    std::cout << "[Host] OCALL: ocall_onnx_load_model received (" << model_data_len << " bytes)." << std::endl;
    // ... (initial parameter checks are the same)

    OrtSessionOptions* session_options = nullptr;
    // ... (CreateSessionOptions is the same)

    OrtSession* session = nullptr;
    OrtStatus* status = g_ort_api->CreateSessionFromArray(g_host_ort_env, model_data, model_data_len, session_options, &session);

    // ... (ReleaseSessionOptions and check status are the same)

    // --- NEW: Dynamically Get Model Input/Output Info ---
    OrtAllocator* allocator;
    g_ort_api->GetAllocatorWithDefaultOptions(&allocator);

    size_t input_count, output_count;
    g_ort_api->SessionGetInputCount(session, &input_count);
    g_ort_api->SessionGetOutputCount(session, &output_count);

    std::cout << "[Host] Model has " << input_count << " input(s) and " << output_count << " output(s)." << std::endl;

    // For this example, we still assume 1 input and 1 output for simplicity of the OCALL interface
    if (input_count != 1 || output_count != 1) {
         std::cerr << "[Host] ERROR: This generic example currently supports only 1 input and 1 output." << std::endl;
         g_ort_api->ReleaseSession(session);
         return OE_INVALID_PARAMETER;
    }

    char* input_name_ptr;
    g_ort_api->SessionGetInputName(session, 0, allocator, &input_name_ptr);
    OrtStringReleaser input_name_releaser(allocator, input_name_ptr);
    std::cout << "[Host] Input Node Name: " << input_name_ptr << std::endl;

    char* output_name_ptr;
    g_ort_api->SessionGetOutputName(session, 0, allocator, &output_name_ptr);
    OrtStringReleaser output_name_releaser(allocator, output_name_ptr);
    std::cout << "[Host] Output Node Name: " << output_name_ptr << std::endl;
    // --- End NEW ---

    uint64_t current_host_handle = g_next_host_session_handle++;
    g_host_onnx_sessions[current_host_handle] = session;
    *host_session_handle_out = current_host_handle;

    std::cout << "[Host] ONNX model loaded by host. Host session handle: " << current_host_handle << std::endl;
    return OE_OK;
}

oe_result_t ocall_onnx_run_inference(
    uint64_t host_session_handle,
    const void* input_data_from_enclave,
    size_t input_len_bytes,
    void* output_data_to_enclave,
    size_t output_buf_len_bytes,
    size_t* actual_output_len_bytes_out) {

    // ... (initial parameter and session handle checks are the same)
    OrtSession* session = g_host_onnx_sessions.at(host_session_handle);

    OrtAllocator* allocator;
    g_ort_api->GetAllocatorWithDefaultOptions(&allocator);

    // --- NEW: Get node names dynamically ---
    char* input_name_ptr;
    g_ort_api->SessionGetInputName(session, 0, allocator, &input_name_ptr);
    OrtStringReleaser input_name_releaser(allocator, input_name_ptr);
    const char* input_node_names[] = {input_name_ptr};

    char* output_name_ptr;
    g_ort_api->SessionGetOutputName(session, 0, allocator, &output_name_ptr);
    OrtStringReleaser output_name_releaser(allocator, output_name_ptr);
    const char* output_node_names[] = {output_name_ptr};
    // --- End NEW ---


    // --- NEW: Get input shape dynamically ---
    OrtTypeInfo* input_type_info;
    g_ort_api->SessionGetInputTypeInfo(session, 0, &input_type_info);
    const OrtTensorTypeAndShapeInfo* input_tensor_info;
    g_ort_api->GetTensorTypeAndShape(input_type_info, &input_tensor_info);
    size_t num_dims;
    g_ort_api->GetDimensionsCount(input_tensor_info, &num_dims);
    std::vector<int64_t> input_node_dims(num_dims);
    g_ort_api->GetDimensions(input_tensor_info, input_node_dims.data(), num_dims);

    // Replace dynamic dimensions (like -1 or 'batch_size') with 1 for this single inference run
    size_t total_elements = 1;
    for(size_t i = 0; i < num_dims; ++i) {
        if(input_node_dims[i] < 0) {
            input_node_dims[i] = 1; // Assuming batch size of 1 for dynamic dimension
        }
        total_elements *= input_node_dims[i];
    }
    g_ort_api->ReleaseTypeInfo(input_type_info);

    if (input_len_bytes != total_elements * sizeof(float)) {
        std::cerr << "[Host] ERROR: Provided input data size (" << input_len_bytes
                  << " bytes) does not match model's required input size ("
                  << total_elements * sizeof(float) << " bytes)." << std::endl;
        return OE_INVALID_PARAMETER;
    }
    // --- End NEW ---

    OrtValue* input_tensor = nullptr;
    // ... (CreateCpuMemoryInfo and CreateTensorWithDataAsOrtValue are largely the same, but use the dynamic shapes/sizes)
    // ...

    // --- Run and process output (largely the same logic) ---
    OrtValue* output_tensor = nullptr;
    g_ort_api->Run(session, nullptr, input_node_names, &input_tensor, 1, output_node_names, 1, &output_tensor);
    // ... (The rest of the function remains the same, as it already processes output dynamically)
    // ...
    return OE_OK;
}
// ... (ocall_onnx_release_session and main function can be kept for testing but will be unused by the Go backend)