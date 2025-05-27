/* enclave/enclave.cpp */
#include <stdio.h>
#include <string.h>
#include <vector>
#include <map> // For managing multiple sessions if needed

// Open Enclave API
#include <openenclave/bits/result.h> 
#include <openenclave/enclave.h>     

// EDL generated trusted header (for ECALLs and OCALL declarations)
#include "enclave_t.h" 

// Logging Macro
#define ENCLAVE_LOG(level, fmt, ...) printf("[" level "] [Enclave] " fmt "\n", ##__VA_ARGS__)

// Structure to hold session information within the enclave,
// primarily the handle provided by the host for its ONNX session.
typedef struct _enclave_ml_session {
    uint64_t host_onnx_session_handle;
    // You could store other enclave-specific metadata here if needed
    // e.g., expected input/output tensor properties if known, model hash, etc.
} enclave_ml_session_t;

// Simple map to manage multiple "sessions" if the host can load multiple models.
// The key is an enclave-generated handle.
static std::map<uint64_t, enclave_ml_session_t> g_enclave_sessions;
static uint64_t g_next_enclave_session_handle = 1; // Simple handle generator

// ECALL Implementations
oe_result_t initialize_enclave_ml_context(
    const unsigned char* model_data,
    size_t model_size,
    uint64_t* enclave_session_handle_out) {
    
    ENCLAVE_LOG("INFO", "ECALL: initialize_enclave_ml_context received.");
    if (!model_data || model_size == 0 || !enclave_session_handle_out) {
        ENCLAVE_LOG("ERROR", "Invalid parameters for initialize_enclave_ml_context.");
        return OE_INVALID_PARAMETER;
    }

    oe_result_t result = OE_FAILURE;
    uint64_t host_session_handle = 0;

    // Make an OCALL to the host to load the ONNX model
    ENCLAVE_LOG("INFO", "OCALL: Requesting host to load ONNX model (%zu bytes).", model_size);
    result = ocall_onnx_load_model(&host_session_handle, model_data, model_size);

    if (result != OE_OK) {
        ENCLAVE_LOG("ERROR", "OCALL ocall_onnx_load_model failed with %s.", oe_result_str(result));
        return result;
    }
    if (host_session_handle == 0) { // Host should return a non-zero handle on success
        ENCLAVE_LOG("ERROR", "Host returned an invalid session handle (0) after loading model.");
        return OE_UNEXPECTED;
    }

    ENCLAVE_LOG("INFO", "Host loaded ONNX model successfully. Host session handle: %lu", (unsigned long)host_session_handle);

    // Create and store enclave-side session information
    enclave_ml_session_t new_session;
    new_session.host_onnx_session_handle = host_session_handle;
    
    uint64_t current_enclave_handle = g_next_enclave_session_handle++;
    g_enclave_sessions[current_enclave_handle] = new_session;
    *enclave_session_handle_out = current_enclave_handle;

    ENCLAVE_LOG("INFO", "Enclave ML context initialized. Enclave session handle: %lu", (unsigned long)current_enclave_handle);
    return OE_OK;
}

oe_result_t enclave_infer(
    uint64_t enclave_session_handle,
    const float* input_data,
    size_t input_data_byte_size,
    float* output_buffer,
    size_t output_buffer_size_bytes,
    size_t* actual_output_size_bytes_out) {

    ENCLAVE_LOG("INFO", "ECALL: enclave_infer received for handle %lu.", (unsigned long)enclave_session_handle);

    if (!input_data || input_data_byte_size == 0 ||
        !output_buffer || output_buffer_size_bytes == 0 ||
        !actual_output_size_bytes_out || enclave_session_handle == 0) {
        ENCLAVE_LOG("ERROR", "Invalid parameters for enclave_infer.");
        return OE_INVALID_PARAMETER;
    }

    auto it = g_enclave_sessions.find(enclave_session_handle);
    if (it == g_enclave_sessions.end()) {
        ENCLAVE_LOG("ERROR", "Invalid enclave session handle: %lu", (unsigned long)enclave_session_handle);
        return OE_NOT_FOUND;
    }

    enclave_ml_session_t* session = &it->second;
    oe_result_t result = OE_FAILURE;

    // Make an OCALL to the host to run inference
    ENCLAVE_LOG("INFO", "OCALL: Requesting host to run inference for host handle %lu.", (unsigned long)session->host_onnx_session_handle);
    result = ocall_onnx_run_inference(
        session->host_onnx_session_handle,
        input_data,                     // Pass input data directly
        input_data_byte_size,
        output_buffer,                  // Pass output buffer directly
        output_buffer_size_bytes,
        actual_output_size_bytes_out);  // Host will fill this

    if (result != OE_OK) {
        ENCLAVE_LOG("ERROR", "OCALL ocall_run_onnx_inference failed with %s.", oe_result_str(result));
        return result;
    }

    ENCLAVE_LOG("INFO", "Host inference successful. Actual output size: %zu bytes.", *actual_output_size_bytes_out);
    return OE_OK;
}

oe_result_t terminate_enclave_ml_context(uint64_t enclave_session_handle) {
    ENCLAVE_LOG("INFO", "ECALL: terminate_enclave_ml_context for handle %lu.", (unsigned long)enclave_session_handle);

    if (enclave_session_handle == 0) {
        ENCLAVE_LOG("ERROR", "Invalid enclave session handle (0) for termination.");
        return OE_INVALID_PARAMETER;
    }
    
    auto it = g_enclave_sessions.find(enclave_session_handle);
    if (it == g_enclave_sessions.end()) {
        ENCLAVE_LOG("ERROR", "Enclave session handle %lu not found for termination.", (unsigned long)enclave_session_handle);
        return OE_NOT_FOUND; // Or OE_OK if you want to be lenient
    }

    enclave_ml_session_t* session = &it->second;
    oe_result_t result = OE_FAILURE;

    // Make an OCALL to the host to release its ONNX session
    ENCLAVE_LOG("INFO", "OCALL: Requesting host to release ONNX session for host handle %lu.", (unsigned long)session->host_onnx_session_handle);
    result = ocall_onnx_release_session(session->host_onnx_session_handle);

    if (result != OE_OK) {
        ENCLAVE_LOG("ERROR", "OCALL ocall_onnx_release_session failed with %s.", oe_result_str(result));
        // Proceed to remove enclave-side session info anyway
    } else {
        ENCLAVE_LOG("INFO", "Host released ONNX session successfully.");
    }

    g_enclave_sessions.erase(it);
    ENCLAVE_LOG("INFO", "Enclave ML context for handle %lu terminated.", (unsigned long)enclave_session_handle);
    return OE_OK; // Return OE_OK even if host-side release had an issue, as enclave side is cleaned.
                  // Or propagate host error if critical.
}
