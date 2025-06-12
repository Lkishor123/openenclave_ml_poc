# openenclave_ml_poc/CMakeLists.txt (Root)
cmake_minimum_required(VERSION 3.16)
project(OpenEnclaveMLPoC CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11) 
set(CMAKE_C_STANDARD_REQUIRED ON)

# --- Build Type Handling ---
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# --- Define Open Enclave Crypto Library ---
set(OE_CRYPTO_LIB "mbedtls" CACHE STRING "Open Enclave Crypto Library (mbedtls or openssl)")
message(STATUS "Using Open Enclave Crypto Library: ${OE_CRYPTO_LIB}")

# --- Find Open Enclave SDK ---
find_package(OpenEnclave CONFIG REQUIRED)

if(NOT OpenEnclave_FOUND)
    message(FATAL_ERROR "OpenEnclave SDK not found by find_package! Check OE_SDK_PATH or CMake prefix paths.")
endif()

message(STATUS "[ROOT] Found OpenEnclave SDK version: ${OpenEnclave_VERSION}")
message(STATUS "  [ROOT] OpenEnclave CMake module directory (OpenEnclave_DIR): ${OpenEnclave_DIR}")

# Check and enforce OpenEnclave_INCLUDE_DIRS
if(NOT OpenEnclave_INCLUDE_DIRS)
    message(WARNING "  [ROOT DIAG] OpenEnclave_INCLUDE_DIRS was NOT set by find_package. Attempting to set manually to /opt/openenclave/include.")
    set(OpenEnclave_INCLUDE_DIRS "/opt/openenclave/include" CACHE PATH "Open Enclave SDK include directory (manual fallback)" FORCE)
    if(NOT EXISTS "${OpenEnclave_INCLUDE_DIRS}/openenclave/enclave.h") # Basic check for validity
        message(FATAL_ERROR "  [ROOT DIAG] Manual fallback for OpenEnclave_INCLUDE_DIRS (${OpenEnclave_INCLUDE_DIRS}) does not seem to contain Open Enclave headers (checked for openenclave/enclave.h). Please verify the path to your Open Enclave SDK include directory.")
    else()
        message(STATUS "  [ROOT DIAG] OpenEnclave_INCLUDE_DIRS manually set to: ${OpenEnclave_INCLUDE_DIRS}")
    endif()
else()
    message(STATUS "  [ROOT] OpenEnclave Include Dirs (OpenEnclave_INCLUDE_DIRS): ${OpenEnclave_INCLUDE_DIRS}")
endif()

# Diagnostic: Check if key OE targets are defined after find_package
if(NOT TARGET openenclave::oehost)
    message(FATAL_ERROR "  [ROOT DIAG] CMake target 'openenclave::oehost' IS NOT defined after find_package. This is a critical issue. Check SDK's CMake files in ${OpenEnclave_DIR}.")
else()
    message(STATUS "  [ROOT DIAG] CMake target 'openenclave::oehost' IS defined.")
endif()
if(NOT TARGET openenclave::oeenclave)
    message(FATAL_ERROR "  [ROOT DIAG] CMake target 'openenclave::oeenclave' IS NOT defined after find_package. This is a critical issue.")
else()
    message(STATUS "  [ROOT DIAG] CMake target 'openenclave::oeenclave' IS defined.")
endif()
if(NOT TARGET openenclave::oeedger8r)
    message(FATAL_ERROR "  [ROOT DIAG] CMake target 'openenclave::oeedger8r' IS NOT defined after find_package.")
else()
    message(STATUS "  [ROOT DIAG] CMake target 'openenclave::oeedger8r' IS defined.")
endif()

# --- Build GGML from external directory ---
set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(GGML_SOURCE_DIR ${EXTERNAL_DIR}/ggml)

if(NOT EXISTS "${GGML_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "GGML source not found at ${GGML_SOURCE_DIR}. Did you run scripts/download_deps.sh?")
endif()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${GGML_SOURCE_DIR} external_ggml_build)

get_target_property(GGML_LIBRARY ggml LOCATION)
set(GLOBAL_GGML_INCLUDE_DIR ${GGML_SOURCE_DIR}/include CACHE INTERNAL "GGML include directory")
set(GLOBAL_GGML_LIBRARY ${GGML_LIBRARY} CACHE INTERNAL "GGML library path")

# --- EDL File Processing ---
set(EDL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/common/enclave.edl) 
set(EDL_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/edl_generated)

add_custom_command(OUTPUT ${EDL_GENERATED_DIR}/.dir_stamp
    COMMAND ${CMAKE_COMMAND} -E make_directory ${EDL_GENERATED_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${EDL_GENERATED_DIR}/.dir_stamp
    COMMENT "Creating EDL generated files directory: ${EDL_GENERATED_DIR}"
    VERBATIM)

set(EDL_TRUSTED_C ${EDL_GENERATED_DIR}/enclave_t.c)
set(EDL_TRUSTED_H ${EDL_GENERATED_DIR}/enclave_t.h)
set(EDL_UNTRUSTED_C ${EDL_GENERATED_DIR}/enclave_u.c)
set(EDL_UNTRUSTED_H ${EDL_GENERATED_DIR}/enclave_u.h)
set(EDL_UNTRUSTED_ARGS_H ${EDL_GENERATED_DIR}/enclave_args.h)

add_custom_command(
    OUTPUT ${EDL_TRUSTED_C} ${EDL_TRUSTED_H} 
    COMMAND openenclave::oeedger8r
            --trusted ${EDL_FILE}
            --trusted-dir ${EDL_GENERATED_DIR}
            --search-path ${OpenEnclave_INCLUDE_DIRS} 
    DEPENDS ${EDL_FILE} ${EDL_GENERATED_DIR}/.dir_stamp
    COMMENT "Running oeedger8r (trusted) on ${EDL_FILE}"
    VERBATIM)

add_custom_command(
    OUTPUT ${EDL_UNTRUSTED_C} ${EDL_UNTRUSTED_H} ${EDL_UNTRUSTED_ARGS_H}
    COMMAND openenclave::oeedger8r
            --untrusted ${EDL_FILE}
            --untrusted-dir ${EDL_GENERATED_DIR}
            --search-path ${OpenEnclave_INCLUDE_DIRS} 
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

# Make generated EDL headers and OE SDK headers available globally
include_directories(${EDL_GENERATED_DIR})
include_directories(${OpenEnclave_INCLUDE_DIRS}) 

# Set variables for subdirectories
set(EDL_TRUSTED_C_PATH ${EDL_TRUSTED_C} CACHE INTERNAL "Path to trusted C for EDL")
set(EDL_UNTRUSTED_C_PATH ${EDL_UNTRUSTED_C} CACHE INTERNAL "Path to untrusted C for EDL")
set(GLOBAL_OpenEnclave_INCLUDE_DIRS ${OpenEnclave_INCLUDE_DIRS} CACHE INTERNAL "OpenEnclave SDK Include Dirs")


# --- Add Subdirectories for Host and Enclave ---
add_subdirectory(enclave)
add_subdirectory(host)

# --- Copy Model to Build Directory ---
# The GGML model is not checked in. Developers should run
# `scripts/download_deps.sh` before building or using the run targets.
set(MODEL_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/model/model.ggml)
set(MODEL_BUILD_DIR ${CMAKE_BINARY_DIR}/model)
set(MODEL_DEST_PATH ${MODEL_BUILD_DIR}/model.ggml)

if(EXISTS ${MODEL_SOURCE_PATH})
    add_custom_command(
        TARGET GenerateEDL POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MODEL_BUILD_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MODEL_SOURCE_PATH} ${MODEL_DEST_PATH}
        COMMENT "Copying GGML model to build directory: ${MODEL_DEST_PATH}"
        VERBATIM)
else()
    message(WARNING "GGML model file not found at ${MODEL_SOURCE_PATH}.")
endif()

message(STATUS "Root ML PoC CMake configuration complete. Build type: ${CMAKE_BUILD_TYPE}")