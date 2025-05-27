# openenclave_ml_poc/enclave/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

set(ENCLAVE_NAME enclave_prod) # Your enclave's target name

# Create the executable target initially without explicit sources.
add_executable(${ENCLAVE_NAME} "")

# Add all source files (existing and generated) to the target.
target_sources(${ENCLAVE_NAME} PRIVATE
    enclave.cpp         # Your main enclave C++ file
    ${EDL_TRUSTED_C}    # Generated trusted EDL C file from root CMakeLists.txt
)

# Link the enclave executable against the required Open Enclave libraries.
# OE_CRYPTO_LIB is set in the root CMakeLists.txt (e.g., to "mbedtls")
target_link_libraries(${ENCLAVE_NAME} PRIVATE
    openenclave::oeenclave
    openenclave::oelibcxx         # For C++ enclave code
    openenclave::oecryptombedtls  # Or openenclave::oecryptoopenssl if OE_CRYPTO_LIB is "openssl"
    # Consider adding openenclave::oecore if specific core symbols are missing,
    # though oeenclave should ideally pull it in.
)

# Include ONNX Runtime headers for the enclave (if enclave uses ONNX C API directly)
# GLOBAL_ONNXRUNTIME_INCLUDE_DIR is set in the root CMakeLists.txt
if(GLOBAL_ONNXRUNTIME_INCLUDE_DIR AND EXISTS "${GLOBAL_ONNXRUNTIME_INCLUDE_DIR}")
    target_include_directories(${ENCLAVE_NAME} PRIVATE ${GLOBAL_ONNXRUNTIME_INCLUDE_DIR})
    message(STATUS "Enclave (${ENCLAVE_NAME}): Added ONNX Runtime include directory: ${GLOBAL_ONNXRUNTIME_INCLUDE_DIR}")
else()
    message(WARNING "Enclave (${ENCLAVE_NAME}): GLOBAL_ONNXRUNTIME_INCLUDE_DIR not set or invalid. ONNX headers might not be found.")
endif()

# Optional: Set the Open Enclave API version
# target_compile_definitions(${ENCLAVE_NAME} PRIVATE OE_API_VERSION=2)

# Ensure that the EDL generation target (GenerateEDL from root CMakeLists.txt)
# is completed before attempting to build this enclave.
add_dependencies(${ENCLAVE_NAME} GenerateEDL)

message(STATUS "Configuring Enclave (using add_executable \"\" and target_sources): ${ENCLAVE_NAME}")
message(STATUS "  Enclave sources added via target_sources: enclave.cpp, ${EDL_TRUSTED_C}")
message(STATUS "  Unsigned enclave executable: ${ENCLAVE_NAME}")

# --- Enclave Signing ---
set(ENCLAVE_CONF_FILE ${CMAKE_CURRENT_SOURCE_DIR}/enclave.conf) # Ensure this file exists in enclave/
set(ENCLAVE_PRIVATE_KEY_FILE ${CMAKE_CURRENT_SOURCE_DIR}/enclave_private.pem) # Ensure this file exists in enclave/

# Optional: Command to generate a demo private key if it doesn't exist.
find_package(OpenSSL)
if(OpenSSL_FOUND AND NOT EXISTS ${ENCLAVE_PRIVATE_KEY_FILE})
    add_custom_command(OUTPUT ${ENCLAVE_PRIVATE_KEY_FILE}
        COMMAND ${OpenSSL_EXECUTABLE} genrsa -out ${ENCLAVE_PRIVATE_KEY_FILE} -3 3072
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating self-signed key pair (enclave_private.pem) for enclave signing (DEMO ONLY, if key not present)"
        VERBATIM)
    message(STATUS "Attempting to generate demo key: ${ENCLAVE_PRIVATE_KEY_FILE} (if it does not exist)")
else()
    if(NOT OpenSSL_FOUND)
        message(WARNING "OpenSSL not found. Cannot generate demo private key. Ensure ${ENCLAVE_PRIVATE_KEY_FILE} exists.")
    endif()
    if(EXISTS ${ENCLAVE_PRIVATE_KEY_FILE})
         message(STATUS "Using existing private key: ${ENCLAVE_PRIVATE_KEY_FILE}")
    else()
         message(WARNING "Private key ${ENCLAVE_PRIVATE_KEY_FILE} not found and OpenSSL not available to generate it (or key generation command not run).")
    endif()
endif()

set(SIGNED_ENCLAVE_FILE ${CMAKE_CURRENT_BINARY_DIR}/${ENCLAVE_NAME}.signed.so)

add_custom_command(OUTPUT ${SIGNED_ENCLAVE_FILE}
    DEPENDS
        ${ENCLAVE_NAME}                 # Depends on the unsigned enclave executable
        ${ENCLAVE_CONF_FILE}            # Depends on the enclave configuration file
        ${ENCLAVE_PRIVATE_KEY_FILE}     # Depends on the private key
    COMMAND openenclave::oesign sign    # Use the imported target
            -e $<TARGET_FILE:${ENCLAVE_NAME}>
            -c ${ENCLAVE_CONF_FILE}
            -k ${ENCLAVE_PRIVATE_KEY_FILE}
            -o ${SIGNED_ENCLAVE_FILE}   # Explicitly specify output file name
    COMMENT "Signing enclave ${ENCLAVE_NAME} -> ${SIGNED_ENCLAVE_FILE}"
    VERBATIM)

add_custom_target(${ENCLAVE_NAME}_signed ALL
    DEPENDS ${SIGNED_ENCLAVE_FILE})

message(STATUS "  Signed enclave will be: ${SIGNED_ENCLAVE_FILE}")
message(STATUS "  Ensure ${ENCLAVE_CONF_FILE} and ${ENCLAVE_PRIVATE_KEY_FILE} exist for signing.")
