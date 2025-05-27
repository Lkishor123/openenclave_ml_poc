# openenclave_ml_poc/host/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

set(HOST_APP_NAME ml_host_prod) # Your host application's target name

# Create the executable target initially without explicit sources.
add_executable(${HOST_APP_NAME} "")

# Add all source files (existing and generated) to the target.
target_sources(${HOST_APP_NAME} PRIVATE
    host.cpp            # Your main host C++ file
    ${EDL_UNTRUSTED_C}  # Generated untrusted EDL C file from root CMakeLists.txt
)

# Link the host executable against required libraries.
# GLOBAL_ONNXRUNTIME_LIBRARY and GLOBAL_ONNXRUNTIME_INCLUDE_DIR are set in root CMakeLists.txt
target_link_libraries(${HOST_APP_NAME} PRIVATE
    openenclave::oehost
    ${GLOBAL_ONNXRUNTIME_LIBRARY} # Link ONNX Runtime library
    stdc++fs                      # For std::filesystem (if used in host.cpp)
)

# Include ONNX Runtime headers for the host
if(GLOBAL_ONNXRUNTIME_INCLUDE_DIR AND EXISTS "${GLOBAL_ONNXRUNTIME_INCLUDE_DIR}")
    target_include_directories(${HOST_APP_NAME} PRIVATE ${GLOBAL_ONNXRUNTIME_INCLUDE_DIR})
    message(STATUS "Host (${HOST_APP_NAME}): Added ONNX Runtime include directory: ${GLOBAL_ONNXRUNTIME_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Host (${HOST_APP_NAME}): GLOBAL_ONNXRUNTIME_INCLUDE_DIR not set or invalid. ONNX headers will not be found.")
endif()

# Ensure that EDL generation is completed before attempting to build this host application.
add_dependencies(${HOST_APP_NAME} GenerateEDL)

message(STATUS "Configuring Host Application (using add_executable \"\" and target_sources): ${HOST_APP_NAME}")
message(STATUS "  Host sources added via target_sources: host.cpp, ${EDL_UNTRUSTED_C}")
message(STATUS "  Host executable: ${HOST_APP_NAME}")

# --- Convenience 'run' Target (Optional) ---
# ENCLAVE_NAME should be consistent with the one set in enclave/CMakeLists.txt (e.g., enclave_prod)
set(ENCLAVE_TARGET_NAME enclave_prod) # Must match ENCLAVE_NAME in enclave/CMakeLists.txt
set(ENCLAVE_BUILD_OUTPUT_DIR "${CMAKE_BINARY_DIR}/enclave")
set(SIGNED_ENCLAVE_FULL_PATH "${ENCLAVE_BUILD_OUTPUT_DIR}/${ENCLAVE_TARGET_NAME}.signed.so")

# This 'run' target assumes your host application takes the model path and enclave path as arguments.
# Adjust the command as per your host.cpp's main() function arguments.
# The MODEL_DEST_PATH is defined in the root CMakeLists.txt.
add_custom_target(run
    COMMAND ./${HOST_APP_NAME} ${MODEL_DEST_PATH} ${SIGNED_ENCLAVE_FULL_PATH}
    DEPENDS ${HOST_APP_NAME} ${ENCLAVE_TARGET_NAME}_signed # Depends on host and signed enclave
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} # Runs from build/host/
    COMMENT "Running ML PoC. Host: ./${HOST_APP_NAME}, Model: ${MODEL_DEST_PATH}, Enclave: ${SIGNED_ENCLAVE_FULL_PATH}"
)

add_custom_target(run_simulate
    COMMAND ./${HOST_APP_NAME} ${MODEL_DEST_PATH} ${SIGNED_ENCLAVE_FULL_PATH} --simulate # Assuming your host handles a --simulate flag
    DEPENDS ${HOST_APP_NAME} ${ENCLAVE_TARGET_NAME}_signed
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ML PoC in SIMULATION mode."
)

message(STATUS "  To run: 'make run' or 'make run_simulate' (after successful build).")
message(STATUS "  Run targets expect signed enclave at: ${SIGNED_ENCLAVE_FULL_PATH} and model at ${MODEL_DEST_PATH}")
