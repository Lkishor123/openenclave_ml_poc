/* enclave/enclave.cpp - FINAL CORRECTED VERSION */
#include <stdio.h>
#include <string.h>
#include <vector>
#include <map> 

// Open Enclave API
#include <openenclave/bits/result.h> 
#include <openenclave/enclave.h>     

#include "enclave_t.h" 

#define ENCLAVE_LOG(level, fmt, ...) printf("[" level "] [Enclave] " fmt "\n", ##__VA_ARGS__)

typedef struct _enclave_ml_session {
    uint64_t host_onnx_session_handle;
} enclave_ml_session_t;

static std::map<uint64_t, enclave_ml_session_t> g_enclave_sessions;
static uint64_t g_next_enclave_session_handle = 1; 

oe_result_t initialize_enclave_ml_context(
    const unsigned char* model_data,
    size_t model_size,
    uint64_t* enclave_session_handle_out) {
    
    ENCLAVE_LOG("INFO", "ECALL: initialize_enclave_ml_context received.");
    if (!model_data || model_size == 0 || !enclave_session_handle_out) {
        ENCLAVE_LOG("ERROR", "Invalid parameters for initialize_enclave_ml_context.");
        return OE_INVALID_PARAMETER;
    }

    oe_result_t ocall_mechanism_status = OE_FAILURE;
    oe_result_t ocall_actual_retval = OE_FAILURE; // This will hold the result from the OCALL's logic
    uint64_t host_session_handle = 0;

    ENCLAVE_LOG("INFO", "OCALL: Requesting host to load ONNX model (%zu bytes).", model_size);
    
    // --- FIX: Call the new OCALL signature correctly ---
    ocall_mechanism_status = ocall_onnx_load_model(
        &ocall_actual_retval,        // 1st arg: pointer to the return value
        &host_session_handle,
        model_data,
        model_size);

    if (ocall_mechanism_status != OE_OK) {
        ENCLAVE_LOG("ERROR", "ocall_onnx_load_model mechanism failed with %s.", oe_result_str(ocall_mechanism_status));
        return ocall_mechanism_status;
    }
    if (ocall_actual_retval != OE_OK) {
        ENCLAVE_LOG("ERROR", "Host-side ocall_onnx_load_model logic failed with %s.", oe_result_str(ocall_actual_retval));
        return ocall_actual_retval;
    }

    if (host_session_handle == 0) { 
        ENCLAVE_LOG("ERROR", "Host returned an invalid session handle (0) after loading model.");
        return OE_UNEXPECTED;
    }

    ENCLAVE_LOG("INFO", "Host loaded ONNX model successfully. Host session handle: %lu", (unsigned long)host_session_handle);

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
    const int64_t* input_data,
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
    oe_result_t ocall_mechanism_status = OE_FAILURE;
    oe_result_t ocall_actual_retval = OE_FAILURE;

    ENCLAVE_LOG("INFO", "OCALL: Requesting host to run inference for host handle %lu.", (unsigned long)session->host_onnx_session_handle);
    
    // --- FIX: Call the new OCALL signature correctly ---
    ocall_mechanism_status = ocall_onnx_run_inference(
        &ocall_actual_retval,
        session->host_onnx_session_handle,
        input_data,
        input_data_byte_size,
        output_buffer,
        output_buffer_size_bytes,
        actual_output_size_bytes_out);

    if (ocall_mechanism_status != OE_OK) {
        ENCLAVE_LOG("ERROR", "ocall_onnx_run_inference mechanism failed with %s.", oe_result_str(ocall_mechanism_status));
        return ocall_mechanism_status;
    }
    if (ocall_actual_retval != OE_OK) {
        ENCLAVE_LOG("ERROR", "Host-side ocall_onnx_run_inference logic failed with %s.", oe_result_str(ocall_actual_retval));
        return ocall_actual_retval;
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
        return OE_NOT_FOUND; 
    }

    enclave_ml_session_t* session = &it->second;
    oe_result_t ocall_mechanism_status = OE_FAILURE;
    oe_result_t ocall_actual_retval = OE_FAILURE;

    ENCLAVE_LOG("INFO", "OCALL: Requesting host to release ONNX session for host handle %lu.", (unsigned long)session->host_onnx_session_handle);
    
    // --- FIX: Call the new OCALL signature correctly ---
    ocall_mechanism_status = ocall_onnx_release_session(
        &ocall_actual_retval,
        session->host_onnx_session_handle);

    if (ocall_mechanism_status != OE_OK) {
        ENCLAVE_LOG("ERROR", "ocall_onnx_release_session mechanism failed with %s.", oe_result_str(ocall_mechanism_status));
    } else if (ocall_actual_retval != OE_OK) {
        ENCLAVE_LOG("ERROR", "Host-side ocall_onnx_release_session logic failed with %s.", oe_result_str(ocall_actual_retval));
    } else {
        ENCLAVE_LOG("INFO", "Host released ONNX session successfully.");
    }

    g_enclave_sessions.erase(it);
    ENCLAVE_LOG("INFO", "Enclave ML context for handle %lu terminated.", (unsigned long)enclave_session_handle);
    return OE_OK;
}