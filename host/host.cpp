/* host/host.cpp */
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept> // For std::runtime_error
#include <iomanip>   // For std::fixed, std::setprecision
#include <filesystem> // For path checking (C++17)
#include <map>       // For host-side ONNX session management

// Open Enclave API
#include <openenclave/host.h>
#include <openenclave/result.h> // For oe_result_str

// ONNX Runtime C API
#include <onnxruntime_c_api.h>

// EDL generated untrusted header (contains OCALL declarations and ECALL stubs)
// This should be "enclave_u.h" if your EDL file is named "enclave.edl"
// and located in the "common/" directory as per your root CMakeLists.txt.
#include "enclave_u.h" 

// Helper macro for checking OE host function results
#define OE_HOST_CHECK(oe_result, function_name) \
    do { \
        if ((oe_result) != OE_OK) { \
            std::cerr << "[Host] Error: " << function_name << " failed with " \
                      << oe_result_str(oe_result) << " (0x" << std::hex << oe_result << std::dec << ")" \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(std::string(function_name) + " failed."); \
        } \
    } while (0)

// --- Host-side ONNX Runtime Globals ---
static OrtEnv* g_host_ort_env = nullptr;
static const OrtApi* g_ort_api = nullptr; 
static std::map<uint64_t, OrtSession*> g_host_onnx_sessions; // Map: host_session_handle -> OrtSession*
static uint64_t g_next_host_session_handle = 1; // Simple handle generator

// Helper to load a file into a byte vector
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

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <path_to_model.onnx> <path_to_enclave.signed.so> [--simulate]" << std::endl;
}

// --- OCALL Implementations ---

// OCALL: Host loads the ONNX model
oe_result_t ocall_onnx_load_model(
    uint64_t* host_session_handle_out,
    const unsigned char* model_data,
    size_t model_data_len) {
    
    std::cout << "[Host] OCALL: ocall_onnx_load_model received (" << model_data_len << " bytes)." << std::endl;
    if (!g_ort_api || !g_host_ort_env || !host_session_handle_out || !model_data || model_data_len == 0) {
        std::cerr << "[Host] ERROR: Invalid parameters for ocall_onnx_load_model." << std::endl;
        if(host_session_handle_out) *host_session_handle_out = 0;
        return OE_INVALID_PARAMETER;
    }

    *host_session_handle_out = 0; // Default to invalid handle
    OrtSessionOptions* session_options = nullptr;
    OrtStatus* status = g_ort_api->CreateSessionOptions(&session_options);
    if (status != nullptr) {
        std::cerr << "[Host] ERROR: CreateSessionOptions failed: " << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return OE_FAILURE;
    }
    // Example: g_ort_api->SetIntraOpNumThreads(session_options, 1); // Configure session options if needed

    OrtSession* session = nullptr;
    status = g_ort_api->CreateSessionFromArray(g_host_ort_env, model_data, model_data_len, session_options, &session);

    if (session_options) g_ort_api->ReleaseSessionOptions(session_options); // Release options object

    if (status != nullptr) {
        std::cerr << "[Host] ERROR: CreateSessionFromArray failed: " << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return OE_FAILURE;
    }

    uint64_t current_host_handle = g_next_host_session_handle++;
    g_host_onnx_sessions[current_host_handle] = session;
    *host_session_handle_out = current_host_handle;

    std::cout << "[Host] ONNX model loaded by host. Host session handle: " << current_host_handle << std::endl;
    return OE_OK;
}

// OCALL: Host runs inference
oe_result_t ocall_onnx_run_inference(
    uint64_t host_session_handle,
    const void* input_data_from_enclave, // const float*
    size_t input_len_bytes,
    void* output_data_to_enclave,      // float*
    size_t output_buf_len_bytes,
    size_t* actual_output_len_bytes_out) {

    std::cout << "[Host] OCALL: ocall_run_onnx_inference for host handle " << host_session_handle << std::endl;
    if (!g_ort_api || !input_data_from_enclave || !output_data_to_enclave || !actual_output_len_bytes_out || host_session_handle == 0) {
         std::cerr << "[Host] ERROR: Invalid parameters for ocall_run_onnx_inference." << std::endl;
        return OE_INVALID_PARAMETER;
    }

    auto it = g_host_onnx_sessions.find(host_session_handle);
    if (it == g_host_onnx_sessions.end()) {
        std::cerr << "[Host] ERROR: Invalid host session handle for inference: " << host_session_handle << std::endl;
        return OE_NOT_FOUND;
    }
    OrtSession* session = it->second;
    if (!session) {
        std::cerr << "[Host] ERROR: Null OrtSession found for host handle: " << host_session_handle << std::endl;
        return OE_UNEXPECTED;
    }

    OrtAllocator* allocator = nullptr; // Declare allocator
    OrtStatus* status = g_ort_api->GetAllocatorWithDefaultOptions(&allocator);
    if (status != nullptr) { 
        std::cerr << "[Host] ERROR: GetAllocatorWithDefaultOptions failed: " << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return OE_FAILURE; 
    }

    // For simple_model.onnx (Identity, input: float[None, 2], output: float[None, 2])
    // These should ideally be queried from the model or passed from enclave if known.
    // For this PoC, we assume the model is simple_model.onnx.
    const char* input_node_names[] = {"input_tensor"};
    const char* output_node_names[] = {"output_tensor"};
    
    size_t num_input_elements = input_len_bytes / sizeof(float);
    // Assuming the simple_model.onnx takes [batch_size, 2]
    if (num_input_elements == 0 || num_input_elements % 2 != 0) { 
         std::cerr << "[Host] ERROR: Input data size (" << input_len_bytes << " bytes) is not suitable for [N, 2] float tensor." << std::endl;
        return OE_INVALID_PARAMETER;
    }
    std::vector<int64_t> input_node_dims = {static_cast<int64_t>(num_input_elements / 2), 2};

    OrtValue* input_tensor = nullptr;
    OrtMemoryInfo* memory_info_cpu = nullptr;
    status = g_ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_cpu);
    if (status != nullptr) { /* handle error */ std::cerr << "[Host] ERROR: CreateCpuMemoryInfo failed." << std::endl; return OE_FAILURE; }

    status = g_ort_api->CreateTensorWithDataAsOrtValue(memory_info_cpu, 
                                                      const_cast<void*>(input_data_from_enclave), 
                                                      input_len_bytes, 
                                                      input_node_dims.data(), input_node_dims.size(), 
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
                                                      &input_tensor);
    g_ort_api->ReleaseMemoryInfo(memory_info_cpu); // Release after use
    if (status != nullptr) {
        std::cerr << "[Host] ERROR: CreateTensorWithDataAsOrtValue failed: " << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return OE_FAILURE;
    }

    OrtValue* output_tensor = nullptr;
    status = g_ort_api->Run(session, nullptr, input_node_names, &input_tensor, 1, output_node_names, 1, &output_tensor);
    
    g_ort_api->ReleaseValue(input_tensor); // Release input tensor

    if (status != nullptr) {
        std::cerr << "[Host] ERROR: ONNX Run failed: " << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return OE_FAILURE;
    }

    if (!output_tensor) {
        std::cerr << "[Host] ERROR: ONNX Run produced null output tensor." << std::endl;
        return OE_UNEXPECTED;
    }

    OrtTensorTypeAndShapeInfo* output_info = nullptr;
    status = g_ort_api->GetTensorTypeAndShape(output_tensor, &output_info);
    if (status != nullptr) { /* handle error */ g_ort_api->ReleaseValue(output_tensor); return OE_FAILURE; }

    size_t output_elements_count;
    status = g_ort_api->GetTensorShapeElementCount(output_info, &output_elements_count);
    if (status != nullptr) { /* handle error */ g_ort_api->ReleaseTensorTypeAndShapeInfo(output_info); g_ort_api->ReleaseValue(output_tensor); return OE_FAILURE; }
    
    g_ort_api->ReleaseTensorTypeAndShapeInfo(output_info); // Release after use

    size_t required_output_bytes = output_elements_count * sizeof(float);
    if (actual_output_len_bytes_out) *actual_output_len_bytes_out = required_output_bytes;

    if (required_output_bytes > output_buf_len_bytes) {
        std::cerr << "[Host] ERROR: Output buffer too small. Needed: " << required_output_bytes << ", Provided: " << output_buf_len_bytes << std::endl;
        g_ort_api->ReleaseValue(output_tensor);
        return OE_BUFFER_TOO_SMALL;
    }

    float* output_data_ptr_onnx = nullptr;
    status = g_ort_api->GetTensorMutableData(output_tensor, (void**)&output_data_ptr_onnx);
    if (status != nullptr) { /* handle error */ g_ort_api->ReleaseValue(output_tensor); return OE_FAILURE; }

    memcpy(output_data_to_enclave, output_data_ptr_onnx, required_output_bytes);
    
    g_ort_api->ReleaseValue(output_tensor); // Release output tensor
    std::cout << "[Host] Inference successful via OCALL. Output bytes: " << required_output_bytes << std::endl;
    return OE_OK;
}

// OCALL: Host releases the ONNX session
oe_result_t ocall_onnx_release_session(uint64_t host_session_handle) {
    std::cout << "[Host] OCALL: ocall_onnx_release_session for host handle " << host_session_handle << std::endl;
    if (!g_ort_api || host_session_handle == 0) {
        return OE_INVALID_PARAMETER;
    }
    auto it = g_host_onnx_sessions.find(host_session_handle);
    if (it != g_host_onnx_sessions.end()) {
        if (it->second) {
            g_ort_api->ReleaseSession(it->second);
        }
        g_host_onnx_sessions.erase(it);
        std::cout << "[Host] ONNX session released for host handle " << host_session_handle << std::endl;
        return OE_OK;
    }
    std::cerr << "[Host] ERROR: Host session handle " << host_session_handle << " not found for release." << std::endl;
    return OE_NOT_FOUND;
}


int main(int argc, char* argv[]) {
    oe_result_t oe_host_result;
    oe_enclave_t* enclave = nullptr;
    int host_app_ret_val = 1; // Default to error
    uint64_t enclave_ml_session_handle = 0; // Handle from enclave for its context

    std::cout << "[Host] Production ML PoC Host Application (OCALL Strategy)." << std::endl;

    if (argc != 3 && argc != 4) { 
        print_usage(argv[0]);
        return 1;
    }

    const std::string model_filepath = argv[1];
    const std::string enclave_filepath = argv[2];
    bool simulate = false;
    if (argc == 4 && std::string(argv[3]) == "--simulate") {
        simulate = true;
        std::cout << "[Host] --simulate flag detected." << std::endl;
    }

    std::cout << "[Host] Model path: " << model_filepath << std::endl;
    std::cout << "[Host] Enclave path: " << enclave_filepath << std::endl;

    if (!std::filesystem::exists(model_filepath)) {
        std::cerr << "[Host] Error: Model file not found at " << model_filepath << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(enclave_filepath)) {
        std::cerr << "[Host] Error: Enclave file not found at " << enclave_filepath << std::endl;
        return 1;
    }

    // --- Initialize Host-side ONNX Runtime ---
    g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION); // ORT_API_VERSION is defined in onnxruntime_c_api.h
    if (!g_ort_api) {
        std::cerr << "[Host] ERROR: Failed to get ONNX Runtime API base." << std::endl;
        return 1;
    }
    OrtStatus* ort_status = g_ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "host_app_ort_env", &g_host_ort_env);
    if (ort_status != nullptr) {
        std::cerr << "[Host] ERROR: Failed to create ONNX Runtime env: " << g_ort_api->GetErrorMessage(ort_status) << std::endl;
        g_ort_api->ReleaseStatus(ort_status);
        return 1;
    }
    std::cout << "[Host] ONNX Runtime environment initialized." << std::endl;

    try {
        uint32_t enclave_flags = OE_ENCLAVE_FLAG_DEBUG; // Keep debug for development
        if (simulate) {
            enclave_flags |= OE_ENCLAVE_FLAG_SIMULATE;
        }
        
        std::cout << "[Host] Creating enclave (" << (enclave_flags & OE_ENCLAVE_FLAG_SIMULATE ? "SIMULATE" : "SGX")
                  << (enclave_flags & OE_ENCLAVE_FLAG_DEBUG ? ", DEBUG" : "") << ")..." << std::endl;

        // The create function name is oe_create_<edl_basename>_enclave.
        // If your EDL file is common/enclave.edl, the basename is "enclave".
        oe_host_result = oe_create_enclave_enclave( 
            enclave_filepath.c_str(),
            OE_ENCLAVE_TYPE_AUTO, 
            enclave_flags,
            nullptr, 0, 
            &enclave);
        OE_HOST_CHECK(oe_host_result, "oe_create_enclave_enclave");
        std::cout << "[Host] Enclave created successfully." << std::endl;

        std::vector<unsigned char> model_buffer = load_file_to_buffer(model_filepath);
        std::cout << "[Host] Model loaded into buffer, size: " << model_buffer.size() << " bytes." << std::endl;

        oe_result_t ecall_ret_status;
        std::cout << "[Host] Calling ECALL: initialize_enclave_ml_context." << std::endl;
        oe_host_result = initialize_enclave_ml_context(
            enclave, 
            &ecall_ret_status, 
            model_buffer.data(), 
            model_buffer.size(),
            &enclave_ml_session_handle); // Enclave returns its handle here
        OE_HOST_CHECK(oe_host_result, "initialize_enclave_ml_context (host call)");
        OE_HOST_CHECK(ecall_ret_status, "initialize_enclave_ml_context (enclave execution)");
        std::cout << "[Host] ECALL initialize_enclave_ml_context successful. Enclave session handle: " << enclave_ml_session_handle << std::endl;

        // Prepare sample input data for inference (for simple_model.onnx)
        std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f, 4.0f}; // Example: Batch size 2, 2 features each
        // For simple_model.onnx, which is Identity [None, 2], if batch_size is 1:
        // std::vector<float> input_tensor_values = {3.14f, -2.71f}; 
        size_t input_data_byte_size = input_tensor_values.size() * sizeof(float);

        std::cout << "[Host] Preparing input tensor: [";
        for(size_t i=0; i < input_tensor_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << input_tensor_values[i] << (i == input_tensor_values.size()-1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        std::vector<float> output_tensor_values(1024); // Sufficiently large buffer
        size_t output_buffer_byte_size = output_tensor_values.size() * sizeof(float);
        size_t actual_output_byte_size = 0;

        std::cout << "[Host] Calling ECALL: enclave_infer." << std::endl;
        oe_host_result = enclave_infer(
            enclave,
            &ecall_ret_status,
            enclave_ml_session_handle, // Pass the enclave's session handle
            input_tensor_values.data(),
            input_data_byte_size,
            output_tensor_values.data(),
            output_buffer_byte_size,
            &actual_output_byte_size
        );
        OE_HOST_CHECK(oe_host_result, "enclave_infer (host call)");
        if (ecall_ret_status == OE_BUFFER_TOO_SMALL) { // Check for OE_BUFFER_TOO_SMALL from enclave
            std::cout << "[Host] Enclave reported output buffer too small. Needed: " << actual_output_byte_size
                      << " bytes, Provided: " << output_buffer_byte_size << " bytes." << std::endl;
            // You might want to resize and retry here, or handle as an error.
            throw std::runtime_error("ECALL_enclave_infer failed: OE_BUFFER_TOO_SMALL reported by enclave");
        }
        OE_HOST_CHECK(ecall_ret_status, "enclave_infer (enclave execution)");
        std::cout << "[Host] ECALL enclave_infer successful." << std::endl;

        std::cout << "[Host] Inference output from enclave (actual bytes: " << actual_output_byte_size << "): [";
        size_t output_elements = actual_output_byte_size / sizeof(float);
        output_elements = std::min(output_elements, output_tensor_values.size()); // Don't read past buffer
        for (size_t i = 0; i < output_elements; ++i) {
            std::cout << std::fixed << std::setprecision(6) << output_tensor_values[i] << (i == output_elements - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        std::cout << "[Host] Calling ECALL: terminate_enclave_ml_context." << std::endl;
        oe_host_result = terminate_enclave_ml_context(enclave, &ecall_ret_status, enclave_ml_session_handle);
        OE_HOST_CHECK(oe_host_result, "terminate_enclave_ml_context (host call)");
        if (ecall_ret_status != OE_OK) {
             std::cerr << "[Host] Warning: terminate_enclave_ml_context (enclave execution) failed with "
                       << oe_result_str(ecall_ret_status) << std::endl;
        } else {
            std::cout << "[Host] ECALL terminate_enclave_ml_context successful." << std::endl;
        }
        
        host_app_ret_val = 0; // Success

    } catch (const std::exception& e) {
        std::cerr << "[Host] Critical Error: " << e.what() << std::endl;
        host_app_ret_val = 1;
    } catch (...) {
        std::cerr << "[Host] Critical Error: Unknown exception occurred." << std::endl;
        host_app_ret_val = 1;
    }

    if (enclave) {
        std::cout << "[Host] Terminating enclave." << std::endl;
        oe_terminate_enclave(enclave); // This function does not return oe_result_t in some SDK versions
    }
    
    // --- Cleanup Host-side ONNX Runtime ---
    // Release any remaining sessions (though ideally done by OCALLs from enclave)
    if (g_ort_api) { // Check if g_ort_api was initialized
        for (auto const& [handle, session_ptr] : g_host_onnx_sessions) {
            if (session_ptr) {
                std::cout << "[Host] Releasing orphaned ONNX session handle " << handle << " during host cleanup." << std::endl;
                g_ort_api->ReleaseSession(session_ptr);
            }
        }
        g_host_onnx_sessions.clear();

        if (g_host_ort_env) {
            g_ort_api->ReleaseEnv(g_host_ort_env);
            g_host_ort_env = nullptr;
            std::cout << "[Host] ONNX Runtime environment released." << std::endl;
        }
    }
    
    std::cout << "[Host] Application finished with exit code " << host_app_ret_val << std::endl;
    return host_app_ret_val;
}
