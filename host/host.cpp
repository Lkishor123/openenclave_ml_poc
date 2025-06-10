/* host/host.cpp - CORRECTED AND IMPROVED */
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

// --- Helper Functions and Macros ---
#define OE_HOST_CHECK(oe_result, function_name) \
    do { \
        if ((oe_result) != OE_OK) { \
            std::cerr << "[Host] Error: " << function_name << " failed with " \
                      << oe_result_str(oe_result) << " (0x" << std::hex << oe_result << std::dec << ")" \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(std::string(function_name) + " failed."); \
        } \
    } while (0)

// NEW: Macro to check ONNX Runtime status
static const OrtApi* g_ort_api = nullptr;
#define ORT_CHECK(ort_status) \
    do { \
        if ((ort_status) != nullptr) { \
            const char* msg = g_ort_api->GetErrorMessage(ort_status); \
            std::cerr << "[Host] ONNX Runtime Error: " << msg << std::endl; \
            g_ort_api->ReleaseStatus(ort_status); \
            throw std::runtime_error(std::string("ONNX Runtime failed: ") + msg); \
        } \
    } while (0)


// --- Host-side ONNX Runtime Globals ---
static OrtEnv* g_host_ort_env = nullptr;
static std::map<uint64_t, OrtSession*> g_host_onnx_sessions;
static uint64_t g_next_host_session_handle = 1;

// --- File Loading ---
std::vector<unsigned char> load_file_to_buffer(const std::string& filepath) {
    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("[Host] File not found: " + filepath);
    }
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("[Host] Failed to open file: " + filepath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("[Host] Failed to read file into buffer: " + filepath);
    }
    return buffer;
}


// NEW: Helper to release ONNX C-style strings safely
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

// --- OCALL Implementations (Modified for generic models and error handling) ---
oe_result_t ocall_onnx_load_model(
    uint64_t* host_session_handle_out,
    const unsigned char* model_data,
    size_t model_data_len) {

    try {
        if (!g_ort_api || !g_host_ort_env || !host_session_handle_out || !model_data || model_data_len == 0) {
            return OE_INVALID_PARAMETER;
        }

        *host_session_handle_out = 0;
        OrtSessionOptions* session_options = nullptr;
        ORT_CHECK(g_ort_api->CreateSessionOptions(&session_options));

        OrtSession* session = nullptr;
        ORT_CHECK(g_ort_api->CreateSessionFromArray(g_host_ort_env, model_data, model_data_len, session_options, &session));

        if (session_options) g_ort_api->ReleaseSessionOptions(session_options);

        uint64_t current_host_handle = g_next_host_session_handle++;
        g_host_onnx_sessions[current_host_handle] = session;
        *host_session_handle_out = current_host_handle;

        std::cout << "[Host] ONNX model loaded by host. Host session handle: " << current_host_handle << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Host] Exception in ocall_onnx_load_model: " << e.what() << std::endl;
        return OE_FAILURE;
    }
    return OE_OK;
}

oe_result_t ocall_onnx_run_inference(
    uint64_t host_session_handle,
    const void* input_data_from_enclave,
    size_t input_len_bytes,
    void* output_data_to_enclave,
    size_t output_buf_len_bytes,
    size_t* actual_output_len_bytes_out) {

    try {
        if (!g_ort_api || !input_data_from_enclave || !output_data_to_enclave || !actual_output_len_bytes_out || host_session_handle == 0) {
            return OE_INVALID_PARAMETER;
        }

        auto it = g_host_onnx_sessions.find(host_session_handle);
        if (it == g_host_onnx_sessions.end()) return OE_NOT_FOUND;
        OrtSession* session = it->second;

        OrtAllocator* allocator;
        ORT_CHECK(g_ort_api->GetAllocatorWithDefaultOptions(&allocator));

        // Get node names dynamically
        char* input_name_ptr;
        ORT_CHECK(g_ort_api->SessionGetInputName(session, 0, allocator, &input_name_ptr));
        OrtStringReleaser input_name_releaser(allocator, input_name_ptr);
        const char* input_node_names[] = {input_name_ptr};

        char* output_name_ptr;
        ORT_CHECK(g_ort_api->SessionGetOutputName(session, 0, allocator, &output_name_ptr));
        OrtStringReleaser output_name_releaser(allocator, output_name_ptr);
        const char* output_node_names[] = {output_name_ptr};

        // Get input shape dynamically
        OrtTypeInfo* type_info;
        ORT_CHECK(g_ort_api->SessionGetInputTypeInfo(session, 0, &type_info));
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ORT_CHECK(g_ort_api->CastTypeInfoToTensorInfo(type_info, &tensor_info)); // Use Cast function

        size_t num_dims;
        ORT_CHECK(g_ort_api->GetDimensionsCount(tensor_info, &num_dims));
        std::vector<int64_t> input_node_dims(num_dims);
        ORT_CHECK(g_ort_api->GetDimensions(tensor_info, input_node_dims.data(), num_dims));
        
        g_ort_api->ReleaseTypeInfo(type_info);

        // This logic assumes a single input tensor of type float for now
        size_t total_elements = 1;
        for(size_t i = 0; i < num_dims; ++i) {
            if(input_node_dims[i] < 0) {
                // For dynamic dims (-1), we deduce the size from the input data length.
                // This simple logic assumes only one dynamic dimension (usually batch size).
                size_t known_dims_product = 1;
                for(size_t j = 0; j < num_dims; ++j) {
                    if (i != j) known_dims_product *= input_node_dims[j];
                }
                input_node_dims[i] = (input_len_bytes / sizeof(float)) / known_dims_product;
            }
            total_elements *= input_node_dims[i];
        }

        if (input_len_bytes != total_elements * sizeof(float)) {
             std::cerr << "[Host] ERROR: Input data size mismatch." << std::endl;
             return OE_INVALID_PARAMETER;
        }

        OrtMemoryInfo* memory_info;
        ORT_CHECK(g_ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
        
        OrtValue* input_tensor = nullptr;
        ORT_CHECK(g_ort_api->CreateTensorWithDataAsOrtValue(
            memory_info, const_cast<void*>(input_data_from_enclave), input_len_bytes,
            input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor));

        g_ort_api->ReleaseMemoryInfo(memory_info);

        OrtValue* output_tensor = nullptr;
        ORT_CHECK(g_ort_api->Run(session, nullptr, input_node_names, (const OrtValue* const*)&input_tensor, 1, output_node_names, 1, &output_tensor));
        g_ort_api->ReleaseValue(input_tensor);

        // Process output (this part was mostly correct)
        OrtTensorTypeAndShapeInfo* output_info;
        ORT_CHECK(g_ort_api->GetTensorTypeAndShape(output_tensor, &output_info));

        size_t output_elements_count;
        ORT_CHECK(g_ort_api->GetTensorShapeElementCount(output_info, &output_elements_count));
        g_ort_api->ReleaseTensorTypeAndShapeInfo(output_info);

        size_t required_output_bytes = output_elements_count * sizeof(float);
        if (actual_output_len_bytes_out) *actual_output_len_bytes_out = required_output_bytes;

        if (required_output_bytes > output_buf_len_bytes) {
            g_ort_api->ReleaseValue(output_tensor);
            return OE_BUFFER_TOO_SMALL;
        }

        float* output_data_ptr_onnx = nullptr;
        ORT_CHECK(g_ort_api->GetTensorMutableData(output_tensor, (void**)&output_data_ptr_onnx));
        memcpy(output_data_to_enclave, output_data_ptr_onnx, required_output_bytes);
        
        g_ort_api->ReleaseValue(output_tensor);

    } catch (const std::exception& e) {
        std::cerr << "[Host] Exception in ocall_onnx_run_inference: " << e.what() << std::endl;
        return OE_FAILURE;
    }
    return OE_OK;
}


oe_result_t ocall_onnx_release_session(uint64_t host_session_handle) {
    if (!g_ort_api || host_session_handle == 0) {
        return OE_INVALID_PARAMETER;
    }
    auto it = g_host_onnx_sessions.find(host_session_handle);
    if (it != g_host_onnx_sessions.end()) {
        if (it->second) {
            g_ort_api->ReleaseSession(it->second);
        }
        g_host_onnx_sessions.erase(it);
        return OE_OK;
    }
    return OE_NOT_FOUND;
}

// The main function can be modified to accept the --use-stdin flag
// as discussed in the previous step, or removed if you only use
// this as a library called by the Go backend.
int main(int argc, char* argv[]) {
    // ... For testing, the original main function can be kept.
    // For production, this will not be the main entry point.
    return 0;
}