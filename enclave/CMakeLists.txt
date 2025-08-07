# openenclave_ml_poc/enclave/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

set(ENCLAVE_NAME enclave_prod)

# Create the executable target initially without explicit sources.
add_executable(${ENCLAVE_NAME} "")

# Add all source files (existing and generated) to the target.
# Use CMAKE_CURRENT_SOURCE_DIR to be explicit about the path for enclave.cpp
# EDL_TRUSTED_C_PATH is set in the root CMakeLists.txt
target_sources(${ENCLAVE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/enclave.cpp
    ${EDL_TRUSTED_C_PATH}
)

# Link the enclave executable against the required Open Enclave libraries.

# Ensure that the Open Enclave include directories are added to this target.
# Explicitly add the Open Enclave include directories to this target.
# This ensures the compiler can find "openenclave/edl/sgx/attestation.h" and other headers.
target_include_directories(${ENCLAVE_NAME} PRIVATE
    ${OpenEnclave_INCLUDE_DIRS}
)


# Link the enclave executable against the required Open Enclave libraries.
target_link_libraries(${ENCLAVE_NAME} PRIVATE
    openenclave::oeenclave
    openenclave::oeattester            # generic dispatch layer
    openenclave::oeattestersgx         # SGX‑ECDSA plug‑in
    openenclave::oelibcxx
    openenclave::oecryptombedtls
    ${GLOBAL_GGML_LIBRARY}
)

# Add Include Directories:
# OpenEnclave_INCLUDE_DIRS should be globally available from root CMakeLists.txt's include_directories()
# GLOBAL_OpenEnclave_INCLUDE_DIRS is also available if needed for direct use.

# GGML headers for the enclave
if(GLOBAL_GGML_INCLUDE_DIR AND EXISTS "${GLOBAL_GGML_INCLUDE_DIR}")
    target_include_directories(${ENCLAVE_NAME} PRIVATE ${GLOBAL_GGML_INCLUDE_DIR})
else()
    message(WARNING "Enclave (${ENCLAVE_NAME}): GLOBAL_GGML_INCLUDE_DIR not properly set or found. GGML headers might not be found if enclave.cpp uses them.")
endif()

add_dependencies(${ENCLAVE_NAME} GenerateEDL)

message(STATUS "Configuring Enclave (using OCALL strategy): ${ENCLAVE_NAME}")
message(STATUS "  Enclave sources added: ${CMAKE_CURRENT_SOURCE_DIR}/enclave.cpp, ${EDL_TRUSTED_C_PATH}")

# --- Enclave Signing ---
set(ENCLAVE_CONF_FILE ${CMAKE_CURRENT_SOURCE_DIR}/enclave.conf)
set(ENCLAVE_PRIVATE_KEY_FILE ${CMAKE_CURRENT_SOURCE_DIR}/enclave_private.pem)

find_package(OpenSSL)
if(OpenSSL_FOUND AND NOT EXISTS ${ENCLAVE_PRIVATE_KEY_FILE})
    add_custom_command(OUTPUT ${ENCLAVE_PRIVATE_KEY_FILE}
        COMMAND ${OpenSSL_EXECUTABLE} genrsa -out ${ENCLAVE_PRIVATE_KEY_FILE} -3 3072
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating self-signed key pair (enclave_private.pem) (DEMO ONLY)"
        VERBATIM)
    message(STATUS "Attempting to generate demo key: ${ENCLAVE_PRIVATE_KEY_FILE}")
else()
    if(NOT OpenSSL_FOUND AND NOT EXISTS ${ENCLAVE_PRIVATE_KEY_FILE})
        message(WARNING "OpenSSL not found AND private key missing. Cannot generate demo key.")
    elseif(EXISTS ${ENCLAVE_PRIVATE_KEY_FILE})
         message(STATUS "Using existing private key: ${ENCLAVE_PRIVATE_KEY_FILE}")
    endif()
endif()

set(SIGNED_ENCLAVE_FILE ${CMAKE_CURRENT_BINARY_DIR}/${ENCLAVE_NAME}.signed.so)

add_custom_command(OUTPUT ${SIGNED_ENCLAVE_FILE}
    DEPENDS ${ENCLAVE_NAME} ${ENCLAVE_CONF_FILE} ${ENCLAVE_PRIVATE_KEY_FILE}
    COMMAND openenclave::oesign sign
            -e $<TARGET_FILE:${ENCLAVE_NAME}> -c ${ENCLAVE_CONF_FILE}
            -k ${ENCLAVE_PRIVATE_KEY_FILE} -o ${SIGNED_ENCLAVE_FILE}
    COMMENT "Signing enclave ${ENCLAVE_NAME} -> ${SIGNED_ENCLAVE_FILE}"
    VERBATIM)

add_custom_target(${ENCLAVE_NAME}_signed ALL DEPENDS ${SIGNED_ENCLAVE_FILE})

message(STATUS "  Signed enclave will be: ${SIGNED_ENCLAVE_FILE}")
