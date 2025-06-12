/* host/host.cpp - FINAL STABLE VERSION (SIMPLIFIED & HARDCODED) */
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <map>
#include <numeric>
#include <sstream>

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

// --- Globals and File Loading ---
static OrtEnv* g_host_ort_env = nullptr;
static std::map<uint64_t, OrtSession*> g_host_onnx_sessions;
static uint64_t g_next_host_session_handle = 1;

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

// --- OCALLs ---
oe_result_t ocall_onnx_load_model(
    uint64_t* host_session_handle_out,
    const unsigned char* model_data,
    size_t model_data_len) {
    try {
        if (!g_ort_api || !g_host_ort_env || !host_session_handle_out || !model_data || model_data_len == 0) return OE_INVALID_PARAMETER;
        *host_session_handle_out = 0;
        OrtSessionOptions* session_options = nullptr;
        ORT_CHECK(g_ort_api->CreateSessionOptions(&session_options));
        // Disable advanced memory features and multi-threading to reduce
        // complexity when running inside an OCALL context. These features
        // were observed to trigger heap corruption on some systems.
        ORT_CHECK(g_ort_api->DisableMemPattern(session_options));
        ORT_CHECK(g_ort_api->DisablePerSessionThreads(session_options));
        ORT_CHECK(g_ort_api->SetIntraOpNumThreads(session_options, 1));
        ORT_CHECK(g_ort_api->SetInterOpNumThreads(session_options, 1));
        OrtSession* session = nullptr;
        ORT_CHECK(g_ort_api->CreateSessionFromArray(g_host_ort_env, model_data, model_data_len, session_options, &session));
        if (session_options) g_ort_api->ReleaseSessionOptions(session_options);
        uint64_t current_host_handle = g_next_host_session_handle++;
        g_host_onnx_sessions[current_host_handle] = session;
        *host_session_handle_out = current_host_handle;
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
        auto it = g_host_onnx_sessions.find(host_session_handle);
        if (it == g_host_onnx_sessions.end()) return OE_NOT_FOUND;
        OrtSession* session = it->second;

        // --- SIMPLIFICATION: Hardcode the known input and output names ---
        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {"logits"};

        // Create a local copy of the input data from the enclave to ensure memory safety
        size_t num_tokens = input_len_bytes / sizeof(int64_t);
        std::vector<int64_t> local_input_ids(num_tokens);
        memcpy(local_input_ids.data(), input_data_from_enclave, input_len_bytes);

        // Prepare tensor data
        std::vector<int64_t> attention_mask_data(num_tokens, 1);
        std::vector<int64_t> input_shape = {1, (int64_t)num_tokens};
        
        OrtMemoryInfo* memory_info;
        // The input buffers are allocated by this host code, not by ONNX Runtime
        // itself. Use the device allocator variant rather than the arena
        // allocator to avoid any attempt by ORT to manage this memory.
        ORT_CHECK(g_ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info));
        
        OrtValue* input_ids_tensor = nullptr;
        ORT_CHECK(g_ort_api->CreateTensorWithDataAsOrtValue(
            memory_info, local_input_ids.data(), input_len_bytes,
            input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_ids_tensor));

        OrtValue* attention_mask_tensor = nullptr;
        ORT_CHECK(g_ort_api->CreateTensorWithDataAsOrtValue(
            memory_info, attention_mask_data.data(), attention_mask_data.size() * sizeof(int64_t),
            input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &attention_mask_tensor));
        
        std::vector<OrtValue*> input_tensors = {input_ids_tensor, attention_mask_tensor};
        OrtValue* output_tensor = nullptr;

        // Run Inference
        ORT_CHECK(g_ort_api->Run(session, nullptr, input_names, input_tensors.data(), 2, output_names, 1, &output_tensor));

        // Process Output
        OrtTensorTypeAndShapeInfo* output_info;
        ORT_CHECK(g_ort_api->GetTensorTypeAndShape(output_tensor, &output_info));
        size_t output_elements_count;
        ORT_CHECK(g_ort_api->GetTensorShapeElementCount(output_info, &output_elements_count));
        size_t required_output_bytes = output_elements_count * sizeof(float);
        if (actual_output_len_bytes_out) *actual_output_len_bytes_out = required_output_bytes;

        if (required_output_bytes <= output_buf_len_bytes) {
            float* output_data_ptr_onnx = nullptr;
            ORT_CHECK(g_ort_api->GetTensorMutableData(output_tensor, (void**)&output_data_ptr_onnx));
            memcpy(output_data_to_enclave, output_data_ptr_onnx, required_output_bytes);
        }
        
        // Cleanup
        g_ort_api->ReleaseTensorTypeAndShapeInfo(output_info);
        g_ort_api->ReleaseValue(output_tensor);
        g_ort_api->ReleaseValue(attention_mask_tensor);
        g_ort_api->ReleaseValue(input_ids_tensor);
        g_ort_api->ReleaseMemoryInfo(memory_info);

        if (required_output_bytes > output_buf_len_bytes) {
            return OE_BUFFER_TOO_SMALL;
        }

    } catch (const std::exception& e) {
        // The ORT_CHECK macro will have already printed the detailed error
        return OE_FAILURE;
    }
    return OE_OK;
}

oe_result_t ocall_onnx_release_session(uint64_t host_session_handle) {
    if (!g_ort_api || host_session_handle == 0) return OE_INVALID_PARAMETER;
    auto it = g_host_onnx_sessions.find(host_session_handle);
    if (it != g_host_onnx_sessions.end()) {
        if (it->second) g_ort_api->ReleaseSession(it->second);
        g_host_onnx_sessions.erase(it);
        return OE_OK;
    }
    return OE_NOT_FOUND;
}

int main(int argc, char* argv[]) {
    oe_result_t oe_host_result;
    oe_enclave_t* enclave = nullptr;
    int host_app_ret_val = 1;
    uint64_t enclave_ml_session_handle = 0;

    if (argc < 3) {
        return 1;
    }

    const std::string model_filepath = argv[1];
    const std::string enclave_filepath = argv[2];
    bool use_stdin = false;
    bool simulate = false;

    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--use-stdin") use_stdin = true;
        else if (std::string(argv[i]) == "--simulate") simulate = true;
    }

    g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ORT_CHECK(g_ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "host_app_ort_env", &g_host_ort_env));

    try {
        uint32_t enclave_flags = OE_ENCLAVE_FLAG_DEBUG;
        if (simulate) enclave_flags |= OE_ENCLAVE_FLAG_SIMULATE;

        OE_HOST_CHECK(oe_create_enclave_enclave(
            enclave_filepath.c_str(), OE_ENCLAVE_TYPE_AUTO,
            enclave_flags, nullptr, 0, &enclave), "oe_create_enclave_enclave");

        std::vector<unsigned char> model_buffer = load_file_to_buffer(model_filepath);
        oe_result_t ecall_ret_status;
        OE_HOST_CHECK(initialize_enclave_ml_context(
            enclave, &ecall_ret_status, model_buffer.data(),
            model_buffer.size(), &enclave_ml_session_handle), "initialize_enclave_ml_context (host call)");
        OE_HOST_CHECK(ecall_ret_status, "initialize_enclave_ml_context (enclave execution)");

        std::vector<int64_t> input_tensor_values;
        if (use_stdin) {
            std::string line;
            std::getline(std::cin, line);
            std::stringstream ss(line);
            std::string value_str;
            while(std::getline(ss, value_str, ',')) {
                input_tensor_values.push_back(std::stoll(value_str));
            }
        }

        size_t input_data_byte_size = input_tensor_values.size() * sizeof(int64_t);
        std::vector<float> output_tensor_values(20);
        size_t output_buffer_byte_size = output_tensor_values.size() * sizeof(float);
        size_t actual_output_byte_size = 0;

        OE_HOST_CHECK(enclave_infer(
            enclave, &ecall_ret_status, enclave_ml_session_handle,
            input_tensor_values.data(), input_data_byte_size,
            output_tensor_values.data(), output_buffer_byte_size, &actual_output_byte_size), "enclave_infer (host call)");
        OE_HOST_CHECK(ecall_ret_status, "enclave_infer (enclave execution)");

        size_t output_elements = actual_output_byte_size / sizeof(float);
        for (size_t i = 0; i < output_elements; ++i) {
            std::cout << output_tensor_values[i] << (i == output_elements - 1 ? "" : ", ");
        }
        std::cout << std::endl;

        terminate_enclave_ml_context(enclave, &ecall_ret_status, enclave_ml_session_handle);
        host_app_ret_val = 0;

    } catch (const std::exception& e) {
        // Error messages are printed by the macros, so we just set the return value
        host_app_ret_val = 1;
    }

    if (enclave) oe_terminate_enclave(enclave);
    if (g_host_ort_env) g_ort_api->ReleaseEnv(g_host_ort_env);
    return host_app_ret_val;
}