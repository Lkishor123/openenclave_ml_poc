# openenclave_ml_poc/CMakeLists.txt (Root)
cmake_minimum_required(VERSION 3.16)
project(OpenEnclaveMLPoC CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11) # Or 99 if preferred for C portions
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
message(STATUS "Found OpenEnclave SDK version: ${OpenEnclave_VERSION}")
message(STATUS "  OpenEnclave CMake module directory: ${OpenEnclave_DIR}")

# --- Find ONNX Runtime ---
option(ONNXRUNTIME_ROOT_DIR "Root directory of the ONNX Runtime prebuilt library and headers" "/opt/onnxruntime")
message(STATUS "Attempting to use ONNXRUNTIME_ROOT_DIR: ${ONNXRUNTIME_ROOT_DIR} (This can be overridden with -D parameter)")

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT_DIR}/include")
set(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_ROOT_DIR}/lib")

if(NOT EXISTS "${ONNXRUNTIME_ROOT_DIR}")
    message(FATAL_ERROR "ONNXRUNTIME_ROOT_DIR '${ONNXRUNTIME_ROOT_DIR}' does not exist. Please ensure ONNX Runtime is installed there or provide a correct path via -DONNXRUNTIME_ROOT_DIR.")
endif()
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
    message(FATAL_ERROR "ONNX Runtime header 'onnxruntime_c_api.h' not found in presumed include directory: ${ONNXRUNTIME_INCLUDE_DIR}. Verify your ONNXRUNTIME_ROOT_DIR.")
endif()

find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_LIBRARY_DIR}" NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_LIBRARY)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_LIBRARY_DIR}") # Try again with default paths
    if(NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "libonnxruntime not found in presumed library directory: ${ONNXRUNTIME_LIBRARY_DIR}. Verify your ONNXRUNTIME_ROOT_DIR.")
    endif()
endif()
message(STATUS "  ONNX Runtime Include Dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "  ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")

# Make ONNX settings available to subdirectories
set(GLOBAL_ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE_DIR} CACHE INTERNAL "ONNX Runtime include directory")
set(GLOBAL_ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_LIBRARY} CACHE INTERNAL "ONNX Runtime library path")

# --- EDL File Processing (common to host and enclave) ---
set(EDL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/common/enclave.edl) # Ensure this path is correct
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
set(EDL_UNTRUSTED_ARGS_H ${EDL_GENERATED_DIR}/enclave_args.h) # oeedger8r generates _args.h

add_custom_command(
    OUTPUT ${EDL_TRUSTED_C} ${EDL_TRUSTED_H} # Add _args.h if needed by trusted code
    COMMAND openenclave::oeedger8r
            --trusted ${EDL_FILE}
            --trusted-dir ${EDL_GENERATED_DIR}
            --search-path /opt/openenclave/include # For system EDL imports
    DEPENDS ${EDL_FILE} ${EDL_GENERATED_DIR}/.dir_stamp
    COMMENT "Running oeedger8r (trusted) on ${EDL_FILE}"
    VERBATIM)

add_custom_command(
    OUTPUT ${EDL_UNTRUSTED_C} ${EDL_UNTRUSTED_H} ${EDL_UNTRUSTED_ARGS_H}
    COMMAND openenclave::oeedger8r
            --untrusted ${EDL_FILE}
            --untrusted-dir ${EDL_GENERATED_DIR}
            --search-path /opt/openenclave/include # For system EDL imports
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

# --- Add Subdirectories for Host and Enclave ---
add_subdirectory(enclave)
add_subdirectory(host)

# --- Copy Model to Build Directory (Optional, for convenience) ---
set(MODEL_SOURCE_PATH ${CMAKE_SOURCE_DIR}/model/simple_model.onnx) # Adjust if your model name is different
set(MODEL_BUILD_DIR ${CMAKE_BINARY_DIR}/model)
set(MODEL_DEST_PATH ${MODEL_BUILD_DIR}/simple_model.onnx) # Adjust if your model name is different

if(EXISTS ${MODEL_SOURCE_PATH})
    # This command runs as part of the GenerateEDL target completion.
    # Consider making a separate target if the host depends on this at config time.
    add_custom_command(
        TARGET GenerateEDL POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MODEL_BUILD_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MODEL_SOURCE_PATH} ${MODEL_DEST_PATH}
        COMMENT "Copying ONNX model to build directory: ${MODEL_DEST_PATH}"
        VERBATIM)
else()
    message(WARNING "ONNX model file not found at ${MODEL_SOURCE_PATH}. Ensure it is generated or placed there (e.g., by running scripts/create_model.py). Host application might fail if it expects it in the build directory.")
endif()

message(STATUS "Root ML PoC CMake configuration complete. Build type: ${CMAKE_BUILD_TYPE}")
