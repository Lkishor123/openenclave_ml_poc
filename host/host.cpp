/* host/host.cpp */
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stdexcept> // For std::runtime_error
#include <iomanip>   // For std::fixed, std::setprecision
#include <filesystem> // For path checking (C++17)

// Open Enclave API
#include <openenclave/host.h>
#include <openenclave/result.h> // For oe_result_str

// EDL generated untrusted header
#include "enclave_u.h" // _u suffix is added by edger8r

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

// Helper to load a file into a byte vector
std::vector<unsigned char> load_file_to_buffer(const std::string& filepath) {
    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("File not found: " + filepath);
    }
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read file into buffer: " + filepath);
    }
    return buffer;
}

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <path_to_model.onnx> <path_to_enclave.signed.so>" << std::endl;
}

int main(int argc, char* argv[]) {
    oe_result_t oe_host_result;
    oe_enclave_t* enclave = nullptr;
    int host_app_ret_val = 1; // Default to error

    std::cout << "[Host] Production ML PoC Host Application." << std::endl;

    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string model_filepath = argv[1];
    const std::string enclave_filepath = argv[2];

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

    try {
        // Create the enclave
        // Use OE_ENCLAVE_FLAG_SIMULATE for simulation mode if SGX hardware is not available/configured.
        // Remove OE_ENCLAVE_FLAG_DEBUG for production builds.
        uint32_t enclave_flags = OE_ENCLAVE_FLAG_DEBUG;
        // uint32_t enclave_flags = 0; // For release SGX hardware
        // uint32_t enclave_flags = OE_ENCLAVE_FLAG_SIMULATE; // For simulation

        std::cout << "[Host] Creating enclave (" << (enclave_flags & OE_ENCLAVE_FLAG_SIMULATE ? "SIMULATE" : "SGX")
                  << (enclave_flags & OE_ENCLAVE_FLAG_DEBUG ? ", DEBUG" : "") << ")..." << std::endl;

        oe_host_result = oe_create_enclave_u(
            enclave_filepath.c_str(),
            OE_ENCLAVE_TYPE_SGX, // Or OE_ENCLAVE_TYPE_AUTO
            enclave_flags,
            nullptr, 0, // settings (e.g., for enclave heap size)
            &enclave);
        OE_HOST_CHECK(oe_host_result, "oe_create_enclave_u");
        std::cout << "[Host] Enclave created successfully." << std::endl;

        // --- Attestation Placeholder ---
        // In a production scenario, the host would now:
        // 1. Trigger the enclave to generate an SGX quote (e.g., via an ECALL like `get_enclave_report` or `get_evidence`).
        // 2. Send this quote to a remote attestation service for verification.
        // 3. Only proceed if attestation is successful.
        // std::cout << "[Host] Placeholder: Perform remote attestation here." << std::endl;


        // Load ONNX model from file
        std::cout << "[Host] Loading ONNX model from " << model_filepath << std::endl;
        std::vector<unsigned char> model_buffer = load_file_to_buffer(model_filepath);
        std::cout << "[Host] Model loaded, size: " << model_buffer.size() << " bytes." << std::endl;

        // --- Secure Model Provisioning Placeholder ---
        // If the model was encrypted, the host (after attesting the enclave) might:
        // 1. Receive a decryption key from the attestation service or a key management service.
        // 2. Pass this key (or a key derived from it) to the enclave via a secure ECALL.
        // 3. The enclave would then decrypt the model_buffer internally.
        // For this example, we pass the model in plaintext.

        // ECALL: Initialize ML model in enclave
        oe_result_t ecall_ret_status; // This will hold the oe_result_t returned by the ECALL function itself
        std::cout << "[Host] Calling ECALL_initialize_enclave_ml." << std::endl;
        oe_host_result = initialize_enclave_ml(enclave, &ecall_ret_status, model_buffer.data(), model_buffer.size());
        OE_HOST_CHECK(oe_host_result, "initialize_enclave_ml (host call)"); // Checks host-side error for the ECALL
        OE_HOST_CHECK(ecall_ret_status, "initialize_enclave_ml (enclave execution)"); // Checks enclave-side execution status
        std::cout << "[Host] ECALL_initialize_enclave_ml successful." << std::endl;


        // Prepare sample input data for inference
        // For the simple_model.onnx (identity, [None, 2] input of floats)
        std::vector<float> input_tensor_values = {3.14f, -2.71f}; // Example: Batch size 1, 2 features
        size_t input_data_byte_size = input_tensor_values.size() * sizeof(float);

        std::cout << "[Host] Preparing input tensor: [";
        for(size_t i=0; i < input_tensor_values.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << input_tensor_values[i] << (i == input_tensor_values.size()-1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        // Prepare output buffer
        // Assume output is same shape as input for identity model.
        // For a production system, you might need an ECALL to query expected output size first,
        // or have a contract for max output size.
        std::vector<float> output_tensor_values(1024); // Reasonably sized buffer, model output should be smaller
        size_t output_buffer_byte_size = output_tensor_values.size() * sizeof(float);
        size_t actual_output_byte_size = 0;

        // ECALL: Perform inference
        std::cout << "[Host] Calling ECALL_enclave_infer." << std::endl;
        oe_host_result = enclave_infer(
            enclave,
            &ecall_ret_status,
            input_tensor_values.data(),
            input_data_byte_size,
            output_tensor_values.data(),
            output_buffer_byte_size,
            &actual_output_byte_size
        );
        OE_HOST_CHECK(oe_host_result, "enclave_infer (host call)");
        if (ecall_ret_status == OE_BUFFER_TOO_SMALL) {
            std::cout << "[Host] Enclave reported output buffer too small. Needed: " << actual_output_byte_size
                      << " bytes, Provided: " << output_buffer_byte_size << " bytes." << std::endl;
            // Potentially resize output_tensor_values and retry if applicable.
            // For this example, we'll treat it as an error to proceed with printing.
            throw std::runtime_error("ECALL_enclave_infer failed: OE_BUFFER_TOO_SMALL");
        }
        OE_HOST_CHECK(ecall_ret_status, "enclave_infer (enclave execution)");
        std::cout << "[Host] ECALL_enclave_infer successful." << std::endl;

        // Print results
        std::cout << "[Host] Inference output from enclave (actual bytes: " << actual_output_byte_size << "): [";
        size_t output_elements = actual_output_byte_size / sizeof(float);
        // Ensure we don't read past the allocated buffer, even if actual_output_byte_size is unexpectedly large
        output_elements = std::min(output_elements, output_tensor_values.size());
        for (size_t i = 0; i < output_elements; ++i) {
            std::cout << std::fixed << std::setprecision(6) << output_tensor_values[i] << (i == output_elements - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        // ECALL: Terminate ML resources in enclave
        std::cout << "[Host] Calling ECALL_terminate_enclave_ml." << std::endl;
        oe_host_result = terminate_enclave_ml(enclave, &ecall_ret_status);
        OE_HOST_CHECK(oe_host_result, "terminate_enclave_ml (host call)");
        // Log non-critical enclave termination error, but proceed to terminate enclave
        if (ecall_ret_status != OE_OK) {
             std::cerr << "[Host] Warning: terminate_enclave_ml (enclave execution) failed with "
                       << oe_result_str(ecall_ret_status) << std::endl;
        } else {
            std::cout << "[Host] ECALL_terminate_enclave_ml successful." << std::endl;
        }
        
        host_app_ret_val = 0; // Indicate success

    } catch (const std::exception& e) {
        std::cerr << "[Host] Critical Error: " << e.what() << std::endl;
        host_app_ret_val = 1;
    } catch (...) {
        std::cerr << "[Host] Critical Error: Unknown exception occurred." << std::endl;
        host_app_ret_val = 1;
    }

    // Terminate the enclave
    if (enclave) {
        std::cout << "[Host] Terminating enclave." << std::endl;
        oe_host_result = oe_terminate_enclave(enclave);
        if (oe_host_result != OE_OK) {
            std::cerr << "[Host] Error: oe_terminate_enclave failed with " << oe_result_str(oe_host_result) << std::endl;
            host_app_ret_val = 1; // Ensure error is propagated
        } else {
            std::cout << "[Host] Enclave terminated successfully." << std::endl;
        }
    }
    
    std::cout << "[Host] Application finished with exit code " << host_app_ret_val << std::endl;
    return host_app_ret_val;
}

