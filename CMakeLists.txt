# CMakeLists.txt (Root)
cmake_minimum_required(VERSION 3.16) # Updated minimum version
project(OpenEnclaveMLProd CXX C)

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

# --- Find Open Enclave SDK ---
# This relies on Open Enclave SDK providing CMake find scripts (e.g., via OE_SDK_PATH or system install)
find_package(openenclave REQUIRED)
message(STATUS "Found OpenEnclave SDK version: ${openenclave_VERSION}")
message(STATUS "  OE Edger8r: ${openenclave_EDGER8R}")
message(STATUS "  OE Sign Tool: ${openenclave_SIGNTOOL}")
message(STATUS "  OE Include Dirs: ${openenclave_INCLUDE_DIRS}")
# message(STATUS "  OE Host Libraries: ${openenclave_host_LIBRARIES}") # Link via target openenclave::host
# message(STATUS "  OE Enclave Libraries: ${openenclave_enclave_LIBRARIES}") # Link via target openenclave::enclave

# --- Find ONNX Runtime ---
# Option for user to specify ONNX Runtime root directory
option(ONNXRUNTIME_ROOT_DIR "Root directory of the ONNX Runtime prebuilt library and headers" "")

if(NOT ONNXRUNTIME_ROOT_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT_DIR is not set. Please provide the path to your ONNX Runtime installation, e.g., -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime-linux-x64-1.15.1")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT_DIR}/include")
set(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_ROOT_DIR}/lib")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
    message(FATAL_ERROR "ONNX Runtime header 'onnxruntime_c_api.h' not found in ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_LIBRARY_DIR}")
if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "libonnxruntime not found in ${ONNXRUNTIME_LIBRARY_DIR}. Ensure it's present (e.g., libonnxruntime.so).")
endif()

message(STATUS "Using ONNXRUNTIME_ROOT_DIR: ${ONNXRUNTIME_ROOT_DIR}")
message(STATUS "  ONNX Runtime Include Dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "  ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")

# Make ONNX settings available to subdirectories by creating imported targets or interface libraries
# For simplicity here, we'll pass variables, but interface libraries are cleaner.
set(GLOBAL_ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE_DIR} CACHE INTERNAL "ONNX Runtime include directory")
set(GLOBAL_ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_LIBRARY} CACHE INTERNAL "ONNX Runtime library path")


# --- EDL File Processing (common to host and enclave) ---
set(EDL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/common/enclave.edl)
set(EDL_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/edl_generated) # Output to build directory

# Ensure the output directory exists before edger8r runs
add_custom_command(OUTPUT ${EDL_GENERATED_DIR}/.dir_stamp # Dummy file to represent directory creation
    COMMAND ${CMAKE_COMMAND} -E make_directory ${EDL_GENERATED_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${EDL_GENERATED_DIR}/.dir_stamp
    COMMENT "Creating EDL generated files directory: ${EDL_GENERATED_DIR}"
    VERBATIM)

set(EDL_TRUSTED_C ${EDL_GENERATED_DIR}/enclave_t.c)
set(EDL_TRUSTED_H ${EDL_GENERATED_DIR}/enclave_t.h)
set(EDL_UNTRUSTED_C ${EDL_GENERATED_DIR}/enclave_u.c)
set(EDL_UNTRUSTED_H ${EDL_GENERATED_DIR}/enclave_u.h)

# Custom command to run edger8r for trusted part
add_custom_command(
    OUTPUT ${EDL_TRUSTED_C} ${EDL_TRUSTED_H}
    COMMAND ${openenclave_EDGER8R} --trusted ${EDL_FILE} --trusted-dir ${EDL_GENERATED_DIR} --search-path ${openenclave_INCLUDE_DIRS}
    DEPENDS ${EDL_FILE} ${EDL_GENERATED_DIR}/.dir_stamp # Depends on EDL file and directory creation
    COMMENT "Running edger8r (trusted) on ${EDL_FILE}"
    VERBATIM)

# Custom command to run edger8r for untrusted part
add_custom_command(
    OUTPUT ${EDL_UNTRUSTED_C} ${EDL_UNTRUSTED_H}
    COMMAND ${openenclave_EDGER8R} --untrusted ${EDL_FILE} --untrusted-dir ${EDL_GENERATED_DIR} --search-path ${openenclave_INCLUDE_DIRS}
    DEPENDS ${EDL_FILE} ${EDL_GENERATED_DIR}/.dir_stamp # Depends on EDL file and directory creation
    COMMENT "Running edger8r (untrusted) on ${EDL_FILE}"
    VERBATIM)

# Add a custom target for EDL generation to ensure it runs before compilation of dependent files
add_custom_target(GenerateEDL ALL
    DEPENDS ${EDL_TRUSTED_C} ${EDL_TRUSTED_H} ${EDL_UNTRUSTED_C} ${EDL_UNTRUSTED_H}
)

# Include directory for generated EDL headers for all targets in this project
include_directories(${EDL_GENERATED_DIR})


# --- Add Subdirectories for Host and Enclave ---
add_subdirectory(enclave)
add_subdirectory(host)

# --- Copy Model to Build Directory (Optional, for convenience) ---
# Assumes model is in ${CMAKE_SOURCE_DIR}/model/simple_model.onnx
set(MODEL_SOURCE_PATH ${CMAKE_SOURCE_DIR}/model/simple_model.onnx)
set(MODEL_BUILD_DIR ${CMAKE_BINARY_DIR}/model) # Copy to build/model/
set(MODEL_DEST_PATH ${MODEL_BUILD_DIR}/simple_model.onnx)

if(EXISTS ${MODEL_SOURCE_PATH})
    # This command will run during the build process, specifically after GenerateEDL completes
    # and before the host target (which might depend on it if paths are relative to build dir) is fully built.
    # A better way might be to make the host target depend on a custom target that copies the model.
    add_custom_command(
        TARGET GenerateEDL POST_BUILD # Execute after GenerateEDL target finishes
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MODEL_BUILD_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MODEL_SOURCE_PATH} ${MODEL_DEST_PATH}
        COMMENT "Copying ONNX model to build directory: ${MODEL_DEST_PATH}"
        VERBATIM)
else()
    message(WARNING "ONNX model file not found at ${MODEL_SOURCE_PATH}. Please ensure it is generated or placed there (e.g., by running scripts/create_simple_model.py).")
endif()

message(STATUS "Root CMake configuration complete. Build type: ${CMAKE_BUILD_TYPE}")
