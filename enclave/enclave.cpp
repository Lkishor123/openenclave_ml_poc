/* enclave/enclave.cpp */
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdexcept> // For std::runtime_error (use with caution in enclave)
#include <algorithm> // For std::min

// Open Enclave API
#include <openenclave/bits/result.h> // For oe_result_t, oe_result_str, OE_OK, OE_FAILURE
                                     // This or enclave.h should also bring in OE_ERROR_ codes.
#include <openenclave/enclave.h>


// ONNX Runtime C API header
// This must be available in the include path during enclave compilation.
#include <onnxruntime_c_api.h>

// EDL generated trusted header
#include "enclave_t.h" // _t suffix is added by edger8r

// --- Globals (Minimize globals in production; consider context structures) ---
static OrtEnv* g_ort_env = nullptr;
static OrtSession* g_ort_session = nullptr;
static OrtAllocator* g_ort_allocator = nullptr; // Default allocator from ONNX Runtime

// Store model input/output names and characteristics
// These are populated during model initialization.
static std::vector<char*> g_input_node_names_alloc;  // Storesstrdup'd names
static std::vector<const char*> g_input_node_names_ptr; // Pointers to above
static std::vector<char*> g_output_node_names_alloc; // Stores strdup'd names
static std::vector<const char*> g_output_node_names_ptr; // Pointers to above

// For simplicity, this PoC assumes one input and its dimensions.
// A more robust solution would handle multiple inputs/outputs dynamically.
static std::vector<int64_t> g_expected_input_dims;
static ONNXTensorElementDataType g_expected_input_type;


// --- Utility Functions ---

// Helper to log messages from the enclave.
#define ENCLAVE_LOG(level, fmt, ...) printf("[" level "] [Enclave] " fmt "\n", ##__VA_ARGS__)

// Helper to check ONNX Runtime status and convert to oe_result_t
static oe_result_t check_and_map_ort_status(const OrtApi* ort_api, OrtStatus* status, const char* operation_name) {
    if (status == nullptr) {
        return OE_OK; // No error
    }

    const char* error_message = ort_api->GetErrorMessage(status);
    ENCLAVE_LOG("ERROR", "%s failed: %s", operation_name, error_message);
    ort_api->ReleaseStatus(status); // Release the status object

    return OE_FAILURE; // Generic failure for ONNX errors
}

// Cleanup ONNX resources
static void cleanup_onnx_resources_internal(const OrtApi* ort_api) {
    if (!ort_api) {
        ENCLAVE_LOG("WARN", "OrtApi not available during cleanup_onnx_resources_internal.");
        g_ort_session = nullptr;
        g_ort_env = nullptr;
        g_ort_allocator = nullptr;
        return;
    }

    if (g_ort_session) {
        ort_api->ReleaseSession(g_ort_session);
        g_ort_session = nullptr;
    }
    if (g_ort_env) {
        ort_api->ReleaseEnv(g_ort_env);
        g_ort_env = nullptr;
    }
    g_ort_allocator = nullptr;

    for (auto p : g_input_node_names_alloc) free(p);
    g_input_node_names_alloc.clear();
    g_input_node_names_ptr.clear();

    for (auto p : g_output_node_names_alloc) free(p);
    g_output_node_names_alloc.clear();
    g_output_node_names_ptr.clear();

    g_expected_input_dims.clear();
}

// --- ECALL Implementations ---

oe_result_t initialize_enclave_ml(const unsigned char* model_data, size_t model_size) {
    ENCLAVE_LOG("INFO", "Initializing ML model...");

    if (!model_data || model_size == 0) {
        ENCLAVE_LOG("ERROR", "Invalid model data provided (null or zero size).");
        return OE_INVALID_PARAMETER;
    }

    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort_api) {
        ENCLAVE_LOG("ERROR", "Failed to get ONNX Runtime API base.");
        return OE_FAILURE;
    }

    OrtStatus* status = nullptr;
    oe_result_t result = OE_FAILURE;
    OrtSessionOptions* session_options = nullptr;

    cleanup_onnx_resources_internal(ort_api);

    status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "enclave_ort_env", &g_ort_env);
    result = check_and_map_ort_status(ort_api, status, "CreateEnv");
    if (result != OE_OK) goto cleanup;

    status = ort_api->GetAllocatorWithDefaultOptions(&g_ort_allocator);
    result = check_and_map_ort_status(ort_api, status, "GetAllocatorWithDefaultOptions");
    if (result != OE_OK) goto cleanup;
    if (!g_ort_allocator) {
        ENCLAVE_LOG("ERROR", "Failed to get default allocator even with OK status.");
        result = OE_UNEXPECTED;
        goto cleanup;
    }

    status = ort_api->CreateSessionOptions(&session_options);
    result = check_and_map_ort_status(ort_api, status, "CreateSessionOptions");
    if (result != OE_OK) goto cleanup;

    status = ort_api->CreateSessionFromArray(g_ort_env, model_data, model_size, session_options, &g_ort_session);
    result = check_and_map_ort_status(ort_api, status, "CreateSessionFromArray");
    if (session_options) {
        ort_api->ReleaseSessionOptions(session_options);
        session_options = nullptr;
    }
    if (result != OE_OK) goto cleanup;

    size_t num_input_nodes;
    status = ort_api->SessionGetInputCount(g_ort_session, &num_input_nodes);
    result = check_and_map_ort_status(ort_api, status, "SessionGetInputCount");
    if (result != OE_OK) goto cleanup;

    if (num_input_nodes == 0) {
        ENCLAVE_LOG("ERROR", "Model has no input nodes.");
        result = OE_INVALID_PARAMETER;
        goto cleanup;
    }
    if (num_input_nodes > 0) {
        char* input_name_alloc = nullptr;
        OrtTypeInfo* typeinfo_input = nullptr;
        const OrtTensorTypeAndShapeInfo* tensor_info_input = nullptr;

        status = ort_api->SessionGetInputName(g_ort_session, 0, g_ort_allocator, &input_name_alloc);
        result = check_and_map_ort_status(ort_api, status, "SessionGetInputName (0)");
        if (result != OE_OK) { /* input_name_alloc is not pushed if this fails */ goto cleanup_input_details_early; }
        g_input_node_names_alloc.push_back(input_name_alloc); // Manage allocation
        g_input_node_names_ptr.push_back(g_input_node_names_alloc.back()); // Point to it

        status = ort_api->SessionGetInputTypeInfo(g_ort_session, 0, &typeinfo_input);
        result = check_and_map_ort_status(ort_api, status, "SessionGetInputTypeInfo (0)");
        if (result != OE_OK) goto cleanup_input_details;

        status = ort_api->CastTypeInfoToTensorInfo(typeinfo_input, &tensor_info_input);
        result = check_and_map_ort_status(ort_api, status, "CastTypeInfoToTensorInfo (Input 0)");
        if (result != OE_OK) { if(typeinfo_input) ort_api->ReleaseTypeInfo(typeinfo_input); goto cleanup_input_details; }

        status = ort_api->GetTensorElementType(tensor_info_input, &g_expected_input_type);
        result = check_and_map_ort_status(ort_api, status, "GetTensorElementType (Input 0)");
        if (result != OE_OK) { if(typeinfo_input) ort_api->ReleaseTypeInfo(typeinfo_input); goto cleanup_input_details; }

        size_t num_dims_input;
        status = ort_api->GetDimensionsCount(tensor_info_input, &num_dims_input);
        result = check_and_map_ort_status(ort_api, status, "GetDimensionsCount (Input 0)");
        if (result != OE_OK) { if(typeinfo_input) ort_api->ReleaseTypeInfo(typeinfo_input); goto cleanup_input_details; }

        g_expected_input_dims.resize(num_dims_input);
        status = ort_api->GetDimensions(tensor_info_input, g_expected_input_dims.data(), num_dims_input);
        // No more check_and_map_ort_status for the last GetDimensions, directly check result
        if (status != nullptr) { // If GetDimensions failed
             result = check_and_map_ort_status(ort_api, status, "GetDimensions (Input 0)"); // This will release status
        }
        
    cleanup_input_details:
        if(typeinfo_input) ort_api->ReleaseTypeInfo(typeinfo_input); // Release typeinfo_input if acquired
        if (result != OE_OK) goto cleanup; // General cleanup if any step failed

        ENCLAVE_LOG("INFO", "Model Input 0: Name=%s, Type=%d, Dims=%zu", g_input_node_names_ptr[0], g_expected_input_type, num_dims_input);
    
    cleanup_input_details_early: // In case SessionGetInputName failed
         if (result != OE_OK) goto cleanup;
    }


    size_t num_output_nodes;
    status = ort_api->SessionGetOutputCount(g_ort_session, &num_output_nodes);
    result = check_and_map_ort_status(ort_api, status, "SessionGetOutputCount");
    if (result != OE_OK) goto cleanup;

    if (num_output_nodes == 0) {
        ENCLAVE_LOG("ERROR", "Model has no output nodes.");
        result = OE_INVALID_PARAMETER;
        goto cleanup;
    }
    if (num_output_nodes > 0) {
        char* output_name_alloc = nullptr;
        status = ort_api->SessionGetOutputName(g_ort_session, 0, g_ort_allocator, &output_name_alloc);
        result = check_and_map_ort_status(ort_api, status, "SessionGetOutputName (0)");
        if (result != OE_OK) goto cleanup;
        g_output_node_names_alloc.push_back(output_name_alloc);
        g_output_node_names_ptr.push_back(g_output_node_names_alloc.back());
        ENCLAVE_LOG("INFO", "Model Output 0: Name=%s", g_output_node_names_ptr[0]);
    }

    ENCLAVE_LOG("INFO", "ONNX model initialized successfully.");
    return OE_OK;

cleanup:
    ENCLAVE_LOG("ERROR", "Failed to initialize ONNX model. Cleaning up resources.");
    if (session_options && ort_api) { // Ensure session_options is released if not already
        ort_api->ReleaseSessionOptions(session_options);
    }
    cleanup_onnx_resources_internal(ort_api); // Clean up partially initialized resources
    return result; // Return the specific error code
}


oe_result_t enclave_infer(
    const float* input_data,
    size_t input_data_size_bytes,
    float* output_buffer,
    size_t output_buffer_size_bytes,
    size_t* actual_output_size_bytes) {

    const OrtApi* ort_api = nullptr;
    oe_result_t result = OE_FAILURE;
    OrtStatus* status = nullptr;
    OrtValue* input_tensor = nullptr;
    OrtValue* output_tensor = nullptr;
    OrtMemoryInfo* memory_info = nullptr;
    OrtTensorTypeAndShapeInfo* output_tensor_info = nullptr;
    ONNXTensorElementDataType output_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t output_element_count = 0;
    size_t required_output_buf_size_bytes = 0;
    float* inferred_output_data_ptr = nullptr;

    if (!g_ort_session) {
        ENCLAVE_LOG("ERROR", "ONNX session not initialized. Call initialize_enclave_ml first.");
        return OE_ERROR_INVALID_STATE;
    }
    if (!input_data || input_data_size_bytes == 0 ||
        !output_buffer || output_buffer_size_bytes == 0 ||
        !actual_output_size_bytes) {
        ENCLAVE_LOG("ERROR", "Invalid parameters for inference (null pointers or zero sizes).");
        return OE_INVALID_PARAMETER;
    }
    if (g_input_node_names_ptr.empty() || g_output_node_names_ptr.empty() || g_expected_input_dims.empty()) {
        ENCLAVE_LOG("ERROR", "Model input/output details not populated during init.");
        return OE_ERROR_INVALID_STATE;
    }
    if (g_expected_input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        ENCLAVE_LOG("ERROR", "This enclave_infer implementation currently only supports FLOAT inputs. Model expects type %d.", g_expected_input_type);
        return OE_INVALID_PARAMETER;
    }

    ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort_api) {
        ENCLAVE_LOG("ERROR", "Failed to get ONNX Runtime API for inference.");
        return OE_FAILURE;
    }

    std::vector<int64_t> current_input_dims = g_expected_input_dims;
    size_t expected_num_input_elements = 1;
    bool dynamic_dim_resolved = false;

    for (size_t i = 0; i < current_input_dims.size(); ++i) {
        if (current_input_dims[i] == -1 || current_input_dims[i] == 0) {
            if (i == 0 && !dynamic_dim_resolved) {
                size_t fixed_elements_per_batch = 1;
                bool multiple_dynamic_dims = false;
                for(size_t j = 1; j < current_input_dims.size(); ++j) {
                    if (current_input_dims[j] <= 0) { multiple_dynamic_dims = true; break; }
                    fixed_elements_per_batch *= current_input_dims[j];
                }
                if (multiple_dynamic_dims || fixed_elements_per_batch == 0) {
                    ENCLAVE_LOG("ERROR", "Model has multiple dynamic input dimensions or non-positive static dim.");
                    result = OE_INVALID_PARAMETER; goto infer_cleanup;
                }
                if ((input_data_size_bytes / sizeof(float)) % fixed_elements_per_batch != 0) {
                    ENCLAVE_LOG("ERROR", "Cannot infer dynamic batch size.");
                    result = OE_INVALID_PARAMETER; goto infer_cleanup;
                }
                current_input_dims[i] = (input_data_size_bytes / sizeof(float)) / fixed_elements_per_batch;
                dynamic_dim_resolved = true;
                ENCLAVE_LOG("INFO", "Inferred dynamic batch size to be: %ld", (long)current_input_dims[i]);
            } else {
                ENCLAVE_LOG("ERROR", "Model has unhandled dynamic input dimension at index %zu or already resolved.", i);
                result = OE_INVALID_PARAMETER; goto infer_cleanup;
            }
        }
        if (current_input_dims[i] > 0) {
             expected_num_input_elements *= current_input_dims[i];
        } else {
            ENCLAVE_LOG("ERROR", "Invalid dimension resolved for input tensor.");
            result = OE_INVALID_PARAMETER; goto infer_cleanup;
        }
    }

    if (input_data_size_bytes != expected_num_input_elements * sizeof(float)) {
        ENCLAVE_LOG("ERROR", "Input data size mismatch for resolved shape.");
        result = OE_INVALID_PARAMETER; goto infer_cleanup;
    }

    status = ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    result = check_and_map_ort_status(ort_api, status, "CreateCpuMemoryInfo");
    if (result != OE_OK) goto infer_cleanup;

    status = ort_api->CreateTensorWithDataAsOrtValue(
        memory_info, const_cast<float*>(input_data), input_data_size_bytes,
        current_input_dims.data(), current_input_dims.size(), g_expected_input_type, &input_tensor);
    if(memory_info) { ort_api->ReleaseMemoryInfo(memory_info); memory_info = nullptr; }
    result = check_and_map_ort_status(ort_api, status, "CreateTensorWithDataAsOrtValue");
    if (result != OE_OK) goto infer_cleanup;

    status = ort_api->Run(g_ort_session, nullptr,
                          g_input_node_names_ptr.data(), &input_tensor, g_input_node_names_ptr.size(),
                          g_output_node_names_ptr.data(), g_output_node_names_ptr.size(), &output_tensor);
    if (input_tensor) { ort_api->ReleaseValue(input_tensor); input_tensor = nullptr; }
    result = check_and_map_ort_status(ort_api, status, "Run (Inference)");
    if (result != OE_OK) goto infer_cleanup;

    if (!output_tensor) {
        ENCLAVE_LOG("ERROR", "Inference run but no output tensor.");
        result = OE_UNEXPECTED; goto infer_cleanup;
    }

    status = ort_api->GetTensorTypeAndShape(output_tensor, &output_tensor_info);
    result = check_and_map_ort_status(ort_api, status, "GetTensorTypeAndShape (Output)");
    if (result != OE_OK) goto infer_cleanup;

    status = ort_api->GetTensorElementType(output_tensor_info, &output_type);
    result = check_and_map_ort_status(ort_api, status, "GetTensorElementType (Output)");
    if (result != OE_OK) goto infer_cleanup; // output_tensor_info will be released in cleanup by its own check

    if (output_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        ENCLAVE_LOG("ERROR", "Only FLOAT outputs supported. Model produced type %d.", output_type);
        result = OE_INVALID_PARAMETER; goto infer_cleanup; // output_tensor_info will be released in cleanup
    }

    status = ort_api->GetTensorShapeElementCount(output_tensor_info, &output_element_count);
    if (output_tensor_info) { ort_api->ReleaseTensorTypeAndShapeInfo(output_tensor_info); output_tensor_info = nullptr; } // Release it here
    result = check_and_map_ort_status(ort_api, status, "GetTensorShapeElementCount (Output)");
    if (result != OE_OK) goto infer_cleanup;

    required_output_buf_size_bytes = output_element_count * sizeof(float);
    *actual_output_size_bytes = required_output_buf_size_bytes;

    if (required_output_buf_size_bytes > output_buffer_size_bytes) {
        ENCLAVE_LOG("WARN", "Output buffer too small. Needed: %zu bytes, Provided: %zu bytes.",
               required_output_buf_size_bytes, output_buffer_size_bytes);
        result = OE_BUFFER_TOO_SMALL; goto infer_cleanup;
    }

    status = ort_api->GetTensorMutableData(output_tensor, (void**)&inferred_output_data_ptr);
    result = check_and_map_ort_status(ort_api, status, "GetTensorMutableData (Output)");
    if (result != OE_OK) goto infer_cleanup;

    memcpy(output_buffer, inferred_output_data_ptr, required_output_buf_size_bytes);
    ENCLAVE_LOG("INFO", "Inference successful. Output size: %zu bytes.", required_output_buf_size_bytes);
    result = OE_OK;

infer_cleanup:
    if (ort_api) { // Ensure ort_api is valid before using it
        if (input_tensor) ort_api->ReleaseValue(input_tensor);
        if (output_tensor) ort_api->ReleaseValue(output_tensor);
        // output_tensor_info should have been released earlier if acquired
        if (output_tensor_info) ort_api->ReleaseTensorTypeAndShapeInfo(output_tensor_info);
        // memory_info should have been released earlier if acquired
        if (memory_info) ort_api->ReleaseMemoryInfo(memory_info);
    }
    return result;
}


oe_result_t terminate_enclave_ml() {
    ENCLAVE_LOG("INFO", "Terminating ML resources...");
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    cleanup_onnx_resources_internal(ort_api);
    ENCLAVE_LOG("INFO", "ONNX resources terminated.");
    return OE_OK;
}

