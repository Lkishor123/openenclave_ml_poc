# openenclave_ml_poc/host/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

set(HOST_APP_NAME ml_host_prod) 

add_executable(${HOST_APP_NAME} "")

# Use CMAKE_CURRENT_SOURCE_DIR to be explicit about the path for host.cpp
# EDL_UNTRUSTED_C_PATH is set in the root CMakeLists.txt
target_sources(${HOST_APP_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/host.cpp 
    ${EDL_UNTRUSTED_C_PATH}                   
)

target_link_libraries(${HOST_APP_NAME} PRIVATE
    openenclave::oehost           
    ${GLOBAL_ONNXRUNTIME_LIBRARY} 
    stdc++fs                      
)

# Include Directories for Host:
# OpenEnclave_INCLUDE_DIRS should be globally available from root CMakeLists.txt's include_directories()
# GLOBAL_OpenEnclave_INCLUDE_DIRS is also available if needed.

# ONNX Runtime headers for the host:
if(GLOBAL_ONNXRUNTIME_INCLUDE_DIR AND EXISTS "${GLOBAL_ONNXRUNTIME_INCLUDE_DIR}")
    target_include_directories(${HOST_APP_NAME} PRIVATE ${GLOBAL_ONNXRUNTIME_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Host (${HOST_APP_NAME}): GLOBAL_ONNXRUNTIME_INCLUDE_DIR not properly set or found. ONNX headers will not be found.")
endif()

add_dependencies(${HOST_APP_NAME} GenerateEDL)

message(STATUS "Configuring Host Application (using OCALL strategy): ${HOST_APP_NAME}")
message(STATUS "  Host sources added: ${CMAKE_CURRENT_SOURCE_DIR}/host.cpp, ${EDL_UNTRUSTED_C_PATH}")

# --- Convenience 'run' Target ---
set(ENCLAVE_TARGET_NAME enclave_prod) 
set(ENCLAVE_BUILD_OUTPUT_DIR "${CMAKE_BINARY_DIR}/enclave") # Resolves to build/enclave/
set(SIGNED_ENCLAVE_FULL_PATH "${ENCLAVE_BUILD_OUTPUT_DIR}/${ENCLAVE_TARGET_NAME}.signed.so")

# Reconstruct the model path explicitly for the run target.
# CMAKE_BINARY_DIR is the top-level build directory (e.g., /home/lk/openenclave_ml_poc/build)
set(MODEL_PATH_FOR_RUN_TARGET "${CMAKE_BINARY_DIR}/model/simple_model.onnx") 

# Get the directory of the ONNX Runtime library
get_filename_component(HOST_ONNX_LIB_DIR ${GLOBAL_ONNXRUNTIME_LIBRARY} DIRECTORY)

message(STATUS "[Host CMake] ONNX Runtime Library Directory for LD_LIBRARY_PATH: ${HOST_ONNX_LIB_DIR}")
message(STATUS "[Host CMake] Model destination path for run target: ${MODEL_PATH_FOR_RUN_TARGET}") # Verify this path

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${HOST_ONNX_LIB_DIR}"
            ./${HOST_APP_NAME} ${MODEL_PATH_FOR_RUN_TARGET} ${SIGNED_ENCLAVE_FULL_PATH}
    DEPENDS ${HOST_APP_NAME} ${ENCLAVE_TARGET_NAME}_signed 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} # Execute from build/host/
    COMMENT "Running ML PoC. Host: ./${HOST_APP_NAME}, Model: ${MODEL_PATH_FOR_RUN_TARGET}, Enclave: ${SIGNED_ENCLAVE_FULL_PATH}"
    VERBATIM 
)

add_custom_target(run_simulate
    COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${HOST_ONNX_LIB_DIR}"
            ./${HOST_APP_NAME} ${MODEL_PATH_FOR_RUN_TARGET} ${SIGNED_ENCLAVE_FULL_PATH} --simulate 
    DEPENDS ${HOST_APP_NAME} ${ENCLAVE_TARGET_NAME}_signed
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} # Execute from build/host/
    COMMENT "Running ML PoC in SIMULATION mode. Model: ${MODEL_PATH_FOR_RUN_TARGET}"
    VERBATIM
)

message(STATUS "  To run: 'make run' or 'make run_simulate' (after successful build).")
