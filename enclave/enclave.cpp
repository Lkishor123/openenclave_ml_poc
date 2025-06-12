/* enclave/enclave.cpp - FINAL, DEFINITIVELY CORRECTED VERSION */
#include <stdio.h>
#include <string.h>
#include <vector>
#include <map> 

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
    
    if (!model_data || model_size == 0 || !enclave_session_handle_out) {
        return OE_INVALID_PARAMETER;
    }

    oe_result_t ocall_mechanism_status;
    oe_result_t ocall_actual_retval;
    uint64_t host_session_handle = 0;

    ocall_mechanism_status = ocall_onnx_load_model(
        &ocall_actual_retval,
        &host_session_handle,
        model_data,
        model_size);

    if (ocall_mechanism_status != OE_OK) return ocall_mechanism_status;
    if (ocall_actual_retval != OE_OK) return ocall_actual_retval;
    if (host_session_handle == 0) return OE_UNEXPECTED;

    enclave_ml_session_t new_session = {host_session_handle};
    uint64_t current_enclave_handle = g_next_enclave_session_handle++;
    g_enclave_sessions[current_enclave_handle] = new_session;
    *enclave_session_handle_out = current_enclave_handle;

    return OE_OK;
}

oe_result_t enclave_infer(
    uint64_t enclave_session_handle,
    const int64_t* input_data,
    size_t input_data_byte_size,
    float* output_buffer,
    size_t output_buffer_size_bytes,
    size_t* actual_output_size_bytes_out) {

    if (!input_data || input_data_byte_size == 0 || !output_buffer || output_buffer_size_bytes == 0 || !actual_output_size_bytes_out || enclave_session_handle == 0) {
        return OE_INVALID_PARAMETER;
    }

    auto it = g_enclave_sessions.find(enclave_session_handle);
    if (it == g_enclave_sessions.end()) {
        return OE_NOT_FOUND;
    }

    enclave_ml_session_t* session = &it->second;
    oe_result_t ocall_mechanism_status;
    oe_result_t ocall_actual_retval;

    // The EDL for `ocall_onnx_run_inference` uses `size` which is in bytes.
    // The C++ code in host.cpp correctly uses input_len_bytes.
    ocall_mechanism_status = ocall_onnx_run_inference(
        &ocall_actual_retval,
        session->host_onnx_session_handle,
        input_data,
        input_data_byte_size,
        output_buffer,
        output_buffer_size_bytes,
        actual_output_size_bytes_out);

    if (ocall_mechanism_status != OE_OK) return ocall_mechanism_status;
    if (ocall_actual_retval != OE_OK) return ocall_actual_retval;

    return OE_OK;
}

oe_result_t terminate_enclave_ml_context(uint64_t enclave_session_handle) {
    if (enclave_session_handle == 0) {
        return OE_INVALID_PARAMETER;
    }
    
    auto it = g_enclave_sessions.find(enclave_session_handle);
    if (it == g_enclave_sessions.end()) {
        return OE_NOT_FOUND; 
    }

    enclave_ml_session_t* session = &it->second;
    oe_result_t ocall_mechanism_status;
    oe_result_t ocall_actual_retval;
    
    ocall_mechanism_status = ocall_onnx_release_session(
        &ocall_actual_retval,
        session->host_onnx_session_handle);

    // We proceed with cleanup even if OCALL fails
    g_enclave_sessions.erase(it);

    if (ocall_mechanism_status != OE_OK) return ocall_mechanism_status;
    if (ocall_actual_retval != OE_OK) return ocall_actual_retval;
    
    return OE_OK;
}