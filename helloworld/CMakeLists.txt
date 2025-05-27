cmake_minimum_required(VERSION 3.16)
project(HelloWorldSample C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11) # Or 14/17 if needed, though C is primary here

# Find Open Enclave SDK
# This relies on sourcing openenclaverc or having OE_SDK_PATH set.
# --- BEGIN DIAGNOSTIC ---
message(STATUS "Attempting to find oeedger8r with find_program...")
find_program(OEEDGER8R_DIRECT_PATH oeedger8r
             HINTS /opt/openenclave/bin $ENV{OE_SDK_PATH}/bin $ENV{OE_SDK_PATH}/tools
             PATHS $ENV{PATH} /usr/bin /usr/local/bin)
if(OEEDGER8R_DIRECT_PATH)
    message(STATUS "  Direct find_program found oeedger8r at: ${OEEDGER8R_DIRECT_PATH}")
else()
    message(WARNING "  Direct find_program COULD NOT find oeedger8r.")
    message(STATUS "    Current CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
    message(STATUS "    Current ENV{PATH}: $ENV{PATH}")
    message(STATUS "    Current ENV{OE_SDK_PATH}: $ENV{OE_SDK_PATH}")
endif()
# --- END DIAGNOSTIC ---

find_package(openenclave REQUIRED) # Existing line
# ... rest of your CMakeLists.txt
find_package(openenclave REQUIRED)
message(STATUS "Found OpenEnclave SDK version: ${openenclave_VERSION}")
message(STATUS "  OE Edger8r: ${openenclave_EDGER8R}")
message(STATUS "  OE Sign Tool: ${openenclave_SIGNTOOL}")
message(STATUS "  OE Include Dirs: ${openenclave_INCLUDE_DIRS}")

if(NOT openenclave_EDGER8R)
    message(FATAL_ERROR "OpenEnclave Edger8r (oeedger8r) not found by CMake. Ensure OE environment is sourced.")
endif()

# --- EDL File Processing ---
set(EDL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/common/helloworld.edl)
set(EDL_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/edl_generated)

# Create directory for generated files
add_custom_command(OUTPUT ${EDL_GENERATED_DIR}/.dir_stamp
    COMMAND ${CMAKE_COMMAND} -E make_directory ${EDL_GENERATED_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${EDL_GENERATED_DIR}/.dir_stamp
    COMMENT "Creating EDL generated files directory: ${EDL_GENERATED_DIR}"
    VERBATIM)

# Generate trusted files (for enclave)
set(EDL_TRUSTED_C ${EDL_GENERATED_DIR}/helloworld_t.c)
set(EDL_TRUSTED_H ${EDL_GENERATED_DIR}/helloworld_t.h)
add_custom_command(
    OUTPUT ${EDL_TRUSTED_C} ${EDL_TRUSTED_H}
    COMMAND ${openenclave_EDGER8R} --trusted ${EDL_FILE}
            --trusted-dir ${EDL_GENERATED_DIR}
            --search-path ${openenclave_INCLUDE_DIRS} # For openenclave/edl/syscall.edl, platform.edl
            # The README example also uses --search-path /opt/openenclave/include/openenclave/edl/sgx
            # This might be needed if platform.edl is SGX specific and not found by default search path
            # For now, rely on openenclave_INCLUDE_DIRS which should contain the necessary paths.
            # If building for SGX explicitly:
            # --search-path ${openenclave_INCLUDE_DIRS}/openenclave/edl/sgx # If such structure exists
    DEPENDS ${EDL_FILE} ${EDL_GENERATED_DIR}/.dir_stamp
    COMMENT "Running oeedger8r (trusted) on ${EDL_FILE}"
    VERBATIM)

# Generate untrusted files (for host)
set(EDL_UNTRUSTED_C ${EDL_GENERATED_DIR}/helloworld_u.c)
set(EDL_UNTRUSTED_H ${EDL_GENERATED_DIR}/helloworld_u.h)
set(EDL_UNTRUSTED_ARGS_H ${EDL_GENERATED_DIR}/helloworld_args.h) # oeedger8r also creates _args.h
add_custom_command(
    OUTPUT ${EDL_UNTRUSTED_C} ${EDL_UNTRUSTED_H} ${EDL_UNTRUSTED_ARGS_H}
    COMMAND ${openenclave_EDGER8R} --untrusted ${EDL_FILE}
            --untrusted-dir ${EDL_GENERATED_DIR}
            --search-path ${openenclave_INCLUDE_DIRS}
            # Similar SGX search path consideration as above if needed
    DEPENDS ${EDL_FILE} ${EDL_GENERATED_DIR}/.dir_stamp
    COMMENT "Running oeedger8r (untrusted) on ${EDL_FILE}"
    VERBATIM)

# Custom target for EDL generation
add_custom_target(GenerateEDL ALL
    DEPENDS ${EDL_TRUSTED_C} ${EDL_TRUSTED_H} ${EDL_UNTRUSTED_C} ${EDL_UNTRUSTED_H} ${EDL_UNTRUSTED_ARGS_H}
)

# Include directory for generated EDL headers for all targets
include_directories(${EDL_GENERATED_DIR})

# Add subdirectories for host and enclave
add_subdirectory(enclave)
add_subdirectory(host)