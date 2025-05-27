# ~/openenclave_ml_poc/helloworld/CMakeLists.txt

cmake_minimum_required(VERSION 3.16)
project(HelloWorldSample C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

# Define the crypto library to be used (as per OE v0.13.0+ breaking change)
set(OE_CRYPTO_LIB "mbedtls" CACHE STRING "Open Enclave Crypto Library (mbedtls or openssl)")
message(STATUS "Using Open Enclave Crypto Library: ${OE_CRYPTO_LIB}")

# Use find_package as per the Open Enclave SDK documentation
find_package(OpenEnclave CONFIG REQUIRED)
message(STATUS "Found OpenEnclave SDK version: ${OpenEnclave_VERSION}")
message(STATUS "  OpenEnclave CMake module directory: ${OpenEnclave_DIR}")

# --- EDL File Processing ---
set(EDL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/common/helloworld.edl) # Ensure this path is correct
set(EDL_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/edl_generated)

add_custom_command(OUTPUT ${EDL_GENERATED_DIR}/.dir_stamp
    COMMAND ${CMAKE_COMMAND} -E make_directory ${EDL_GENERATED_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${EDL_GENERATED_DIR}/.dir_stamp
    COMMENT "Creating EDL generated files directory: ${EDL_GENERATED_DIR}"
    VERBATIM)

set(EDL_TRUSTED_C ${EDL_GENERATED_DIR}/helloworld_t.c)
set(EDL_TRUSTED_H ${EDL_GENERATED_DIR}/helloworld_t.h)
set(EDL_UNTRUSTED_C ${EDL_GENERATED_DIR}/helloworld_u.c)
set(EDL_UNTRUSTED_H ${EDL_GENERATED_DIR}/helloworld_u.h)
set(EDL_UNTRUSTED_ARGS_H ${EDL_GENERATED_DIR}/helloworld_args.h)

add_custom_command(
    OUTPUT ${EDL_TRUSTED_C} ${EDL_TRUSTED_H}
    COMMAND openenclave::oeedger8r
            --trusted ${EDL_FILE}
            --trusted-dir ${EDL_GENERATED_DIR}
            --search-path /opt/openenclave/include # Added for system EDL imports
    DEPENDS ${EDL_FILE} ${EDL_GENERATED_DIR}/.dir_stamp
    COMMENT "Running oeedger8r (trusted) on ${EDL_FILE}"
    VERBATIM)

add_custom_command(
    OUTPUT ${EDL_UNTRUSTED_C} ${EDL_UNTRUSTED_H} ${EDL_UNTRUSTED_ARGS_H}
    COMMAND openenclave::oeedger8r
            --untrusted ${EDL_FILE}
            --untrusted-dir ${EDL_GENERATED_DIR}
            --search-path /opt/openenclave/include # Added for system EDL imports
    DEPENDS ${EDL_FILE} ${EDL_GENERATED_DIR}/.dir_stamp
    COMMENT "Running oeedger8r (untrusted) on ${EDL_FILE}"
    VERBATIM)

set_source_files_properties(
    ${EDL_TRUSTED_C} ${EDL_TRUSTED_H}
    ${EDL_UNTRUSTED_C} ${EDL_UNTRUSTED_H} ${EDL_UNTRUSTED_ARGS_H}
    PROPERTIES GENERATED TRUE
)

add_custom_target(GenerateEDL ALL
    DEPENDS ${EDL_TRUSTED_C} ${EDL_TRUSTED_H} ${EDL_UNTRUSTED_C} ${EDL_UNTRUSTED_H} ${EDL_UNTRUSTED_ARGS_H}
)

include_directories(${EDL_GENERATED_DIR})

add_subdirectory(enclave)
add_subdirectory(host)

message(STATUS "Root helloworld CMake configuration complete.")
