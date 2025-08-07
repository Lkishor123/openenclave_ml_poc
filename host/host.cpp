#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <map>
#include <sstream>
#include <cstring>
#include <unistd.h>

#include <openenclave/host.h>
#include <openenclave/bits/result.h>
#include "bert.h"
#include "enclave_u.h"
#include <openenclave/host_verify.h>          // declares oe_free_evidence()

// --- NEW INCLUDES for Attestation ---
#include <iomanip> // For std::hex


#define OE_HOST_CHECK(oe_result, fn) \
    do { \
        if ((oe_result) != OE_OK) { \
            std::cerr << "[Host] " << fn << " failed with " << oe_result_str(oe_result) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(std::string(fn) + " failed"); \
        } \
    } while (0)

static std::map<uint64_t, bert_ctx*> g_sessions;
static uint64_t g_next_session_handle = 1;
static std::string g_model_path;
// Set when the model is loaded to size output tensors appropriately
static int g_embedding_dim = 0;


// Helper function to convert a byte buffer to a hex string for printing
std::string to_hex_string(const unsigned char* buffer, size_t size) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < size; ++i) {
        ss << std::setw(2) << static_cast<unsigned>(buffer[i]);
    }
    return ss.str();
}

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

oe_result_t ocall_ggml_load_model(
    oe_result_t* ocall_host_ret,
    oe_result_t* host_return_value,
    uint64_t* host_session_handle_out,
    const unsigned char* /*model_data*/,
    size_t /*model_data_len*/)
{
    if (!ocall_host_ret || !host_return_value || !host_session_handle_out)
        return OE_INVALID_PARAMETER;

    *ocall_host_ret = OE_OK;
    *host_return_value = OE_FAILURE;
    *host_session_handle_out = 0;

    bert_ctx* ctx = bert_load_from_file(g_model_path.c_str(), true);
    if (!ctx)
        return OE_OK;

    // Capture the embedding dimension from this model
    g_embedding_dim = bert_n_embd(ctx);
    bert_allocate_buffers(ctx, bert_n_max_tokens(ctx), 1);
    uint64_t handle = g_next_session_handle++;
    g_sessions[handle] = ctx;
    *host_session_handle_out = handle;
    *host_return_value = OE_OK;
    return OE_OK;
}

oe_result_t ocall_ggml_run_inference(
    oe_result_t* ocall_host_ret,
    oe_result_t* host_return_value,
    uint64_t host_session_handle,
    const void* input_data_from_enclave,
    size_t input_len_bytes,
    void* output_data_to_enclave,
    size_t output_buf_len_bytes,
    size_t* actual_output_len_bytes_out)
{
    if (!ocall_host_ret || !host_return_value)
        return OE_INVALID_PARAMETER;

    *ocall_host_ret = OE_OK;
    *host_return_value = OE_FAILURE;

    auto it = g_sessions.find(host_session_handle);
    if (it == g_sessions.end()) {
        *host_return_value = OE_NOT_FOUND;
        return OE_OK;
    }

    bert_ctx* ctx = it->second;
    size_t num_tokens = input_len_bytes / sizeof(int64_t);
    const int64_t* tokens64 = static_cast<const int64_t*>(input_data_from_enclave);
    bert_tokens tokens;
    tokens.reserve(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i) {
        tokens.push_back(static_cast<bert_token>(tokens64[i]));
    }

    int n_embd = bert_n_embd(ctx);
    std::vector<float> embeddings(n_embd);
    bert_forward(ctx, tokens, embeddings.data(), 1);

    size_t required = embeddings.size() * sizeof(float);
    if (actual_output_len_bytes_out)
        *actual_output_len_bytes_out = required;

    if (required <= output_buf_len_bytes) {
        memcpy(output_data_to_enclave, embeddings.data(), required);
        *host_return_value = OE_OK;
    } else {
        *host_return_value = OE_BUFFER_TOO_SMALL;
    }
    return OE_OK;
}

oe_result_t ocall_ggml_release_session(
    oe_result_t* ocall_host_ret,
    oe_result_t* host_return_value,
    uint64_t host_session_handle)
{
    if (!ocall_host_ret || !host_return_value)
        return OE_INVALID_PARAMETER;

    *ocall_host_ret = OE_OK;
    *host_return_value = OE_FAILURE;

    auto it = g_sessions.find(host_session_handle);
    if (it != g_sessions.end()) {
        bert_free(it->second);
        g_sessions.erase(it);
        *host_return_value = OE_OK;
    } else {
        *host_return_value = OE_NOT_FOUND;
    }
    return OE_OK;
}

int main(int argc, char* argv[]) {
    oe_enclave_t* enclave = nullptr;
    int host_app_ret_val = 1;
    uint64_t enclave_ml_session_handle = 0;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <enclave_path> [--use-stdin | --attest | --simulate]" << std::endl;
        return 1;
    }
    g_model_path = argv[1];
    const std::string enclave_filepath = argv[2];
    bool use_stdin = false;
    bool simulate = false;
    bool do_attest = false; // New flag for attestation

    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--use-stdin") use_stdin = true;
        else if (std::string(argv[i]) == "--simulate") simulate = true;
        else if (std::string(argv[i]) == "--attest") do_attest = true;
    }

    try {
        uint32_t enclave_flags = OE_ENCLAVE_FLAG_DEBUG;
        if (simulate) enclave_flags |= OE_ENCLAVE_FLAG_SIMULATE;
        OE_HOST_CHECK(oe_create_enclave_enclave(
            enclave_filepath.c_str(), OE_ENCLAVE_TYPE_AUTO,
            enclave_flags, nullptr, 0, &enclave), "oe_create_enclave_enclave");

        // --- ATTESTATION LOGIC ---
        if (do_attest) {
            unsigned char* evidence_buffer = NULL;
            size_t evidence_size = 0;
            bool success = false;
            oe_result_t ecall_result;

            ecall_result = get_attestation_evidence(enclave, &success, &evidence_buffer, &evidence_size);
            OE_HOST_CHECK(ecall_result, "get_attestation_evidence");

            if (!success) {
                 throw std::runtime_error("ECALL to get_attestation_evidence failed.");
            }

            // Print hex string to stdout for the Go app to capture
            std::cout << to_hex_string(evidence_buffer, evidence_size) << std::endl;
            oe_free_evidence(evidence_buffer);
            host_app_ret_val = 0; // Success

        // --- INFERENCE LOGIC (Unchanged) ---
        } else if (use_stdin) {
            std::vector<unsigned char> model_buffer = load_file_to_buffer(g_model_path);
            oe_result_t ecall_ret_status;
            OE_HOST_CHECK(initialize_enclave_ml_context(
                enclave, &ecall_ret_status, model_buffer.data(),
                model_buffer.size(), &enclave_ml_session_handle), "initialize_enclave_ml_context");
            OE_HOST_CHECK(ecall_ret_status, "initialize_enclave_ml_context (enclave)");

            std::string line;
            while (std::getline(std::cin, line)) {
                if (line == "quit" || line == "exit")
                    break;
                if (line.empty())
                    continue;

                std::vector<int64_t> input_tensor_values;
                std::stringstream ss(line);
                std::string value_str;
                while (std::getline(ss, value_str, ',')) {
                    if (!value_str.empty()) {
                        input_tensor_values.push_back(std::stoll(value_str));
                    }
                }

                size_t input_data_byte_size = input_tensor_values.size() * sizeof(int64_t);
                // Allocate buffer based on the model's embedding dimension
                std::vector<float> output_tensor_values(g_embedding_dim);
                size_t output_buffer_byte_size = output_tensor_values.size() * sizeof(float);
                size_t actual_output_byte_size = 0;
                OE_HOST_CHECK(enclave_infer(
                    enclave, &ecall_ret_status, enclave_ml_session_handle,
                    input_tensor_values.data(), input_data_byte_size,
                    output_tensor_values.data(), output_buffer_byte_size,
                    &actual_output_byte_size), "enclave_infer");
                OE_HOST_CHECK(ecall_ret_status, "enclave_infer (enclave)");
                size_t output_elements = actual_output_byte_size / sizeof(float);
                for (size_t i = 0; i < output_elements; ++i) {
                    std::cout << output_tensor_values[i] << (i == output_elements - 1 ? "" : ", ");
                }
                std::cout << std::endl;
            }
            host_app_ret_val = 0;
        }


    } catch (const std::exception& e) {
        std::cerr << "Host exception: " << e.what() << std::endl;
        host_app_ret_val = 1;
    }

        // Tear down the enclave ML context if it was initialized.
    if (enclave_ml_session_handle != 0) {
        oe_result_t ecall_ret_status = OE_FAILURE;
        oe_result_t result = terminate_enclave_ml_context(
            enclave, &ecall_ret_status, enclave_ml_session_handle);
        if (result != OE_OK || ecall_ret_status != OE_OK) {
            std::cerr << "[Host] terminate_enclave_ml_context failed with "
                      << oe_result_str(result != OE_OK ? result : ecall_ret_status)
                      << std::endl;
        }
    }

    if (enclave) oe_terminate_enclave(enclave);

    return host_app_ret_val;
}
