/* enclave/enclave.cpp - Updated with Attestation */
#include <stdio.h>
#include <string.h>
#include <vector>
#include <map>

#include <openenclave/bits/result.h>
#include <openenclave/enclave.h>
#include "enclave_t.h"

// --- NEW INCLUDES for Attestation ---
#include <openenclave/attestation/attester.h>      // oe_get_evidence / oe_attester_initialize
#include <openenclave/attestation/sgx/evidence.h>  // SGX‑specific UUIDs & claims

#define ENCLAVE_LOG(level, fmt, ...) printf("[" level "] [Enclave] " fmt "\n", ##__VA_ARGS__)

typedef struct _enclave_ml_session {
    uint64_t host_ggml_session_handle;
} enclave_ml_session_t;

static std::map<uint64_t, enclave_ml_session_t> g_enclave_sessions;
static uint64_t g_next_enclave_session_handle = 1;

// --- Existing Functions (Unchanged) ---

oe_result_t initialize_enclave_ml_context(
    const unsigned char* model_data,
    size_t model_size,
    uint64_t* enclave_session_handle_out) {

    if (!model_data || model_size == 0 || !enclave_session_handle_out) {
        return OE_INVALID_PARAMETER;
    }

    oe_result_t ocall_status;
    oe_result_t ocall_retval = OE_FAILURE;
    oe_result_t ocall_host_ret = OE_FAILURE;
    oe_result_t host_return_value = OE_FAILURE;
    uint64_t host_session_handle = 0;

    ocall_status = ocall_ggml_load_model(
        &ocall_retval,
        &ocall_host_ret,
        &host_return_value,
        &host_session_handle,
        model_data,
        model_size);
    if (ocall_status != OE_OK) return ocall_status;
    if (ocall_host_ret != OE_OK) return ocall_host_ret;
    if (host_return_value != OE_OK) return host_return_value;
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
    oe_result_t ocall_status;
    oe_result_t ocall_retval = OE_FAILURE;
    oe_result_t ocall_host_ret = OE_FAILURE;
    oe_result_t host_return_value = OE_FAILURE;

    ocall_status = ocall_ggml_run_inference(
        &ocall_retval,
        &ocall_host_ret,
        &host_return_value,
        session->host_ggml_session_handle,
        input_data,
        input_data_byte_size,
        output_buffer,
        output_buffer_size_bytes,
        actual_output_size_bytes_out);
    if (ocall_status != OE_OK) return ocall_status;
    if (ocall_host_ret != OE_OK) return ocall_host_ret;
    if (host_return_value != OE_OK) return host_return_value;

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
    oe_result_t ocall_status;
    oe_result_t ocall_retval = OE_FAILURE;
    oe_result_t ocall_host_ret = OE_FAILURE;
    oe_result_t host_return_value = OE_FAILURE;

    ocall_status = ocall_ggml_release_session(
        &ocall_retval,
        &ocall_host_ret,
        &host_return_value,
        session->host_ggml_session_handle);

    g_enclave_sessions.erase(it);

    if (ocall_status != OE_OK) return ocall_status;
    if (ocall_host_ret != OE_OK) return ocall_host_ret;
    if (host_return_value != OE_OK) return host_return_value;

    return OE_OK;
}

// --- NEW ATTESTATION FUNCTION ---
bool get_attestation_evidence(unsigned char** evidence_buffer, size_t* evidence_size)
{
    if (!evidence_buffer || !evidence_size)
        return false;

    *evidence_buffer = nullptr;
    *evidence_size   = 0;

    // 1) Initialize attester (idempotent on OE 0.19)
    oe_result_t r = oe_attester_initialize();
    if (r != OE_OK)
        return false;

    // 2) Let OE choose a supported format (prefer ECDSA, else local)
    const oe_uuid_t preferred[] = {
        OE_FORMAT_UUID_SGX_ECDSA,             // remote attestation (DCAP)
        OE_FORMAT_UUID_SGX_LOCAL_ATTESTATION  // local (works in simulation)
    };
    oe_uuid_t selected = {};  // output
    r = oe_attester_select_format(preferred,
                                  sizeof(preferred)/sizeof(preferred[0]),
                                  &selected);
    if (r != OE_OK)
        return false;

    // 3) No custom claims / endorsements in this simple flow
    r = oe_get_evidence(&selected,
                        0,
                        nullptr, 0,         // custom claims
                        nullptr, 0,         // endorsements in
                        evidence_buffer,    // out evidence
                        evidence_size,      // out size
                        nullptr, nullptr);  // no endorsements out
    if (r != OE_OK)
        return false;

    return true;
}
