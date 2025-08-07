# openenclave_ml_poc/host/CMakeLists.txt (Modified)
cmake_minimum_required(VERSION 3.16)

# 1. Renamed the host application for clarity
set(HOST_APP_NAME ml_host_prod_go)

add_executable(${HOST_APP_NAME} "")

# Use CMAKE_CURRENT_SOURCE_DIR to be explicit about the path for host.cpp
# EDL_UNTRUSTED_C_PATH is set in the root CMakeLists.txt
target_sources(${HOST_APP_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/host.cpp
    ${EDL_UNTRUSTED_C_PATH}
)

# Link against both bert and ggml. This ensures that the host
# has access to both libraries and, crucially, their include paths.
target_link_libraries(${HOST_APP_NAME} PRIVATE
    openenclave::oehost
    openenclave::oeverifier            # generic dispatch layer
    openenclave::oeverifiersgx         # SGX‑ECDSA plug‑in    bert
    ggml
    stdc++fs
)

# Explicitly add the include directories for bert.cpp and its ggml submodule.
# This ensures the compiler can find "bert.h" and "ggml.h".
target_include_directories(${HOST_APP_NAME} PRIVATE
    ${GLOBAL_BERTCPP_INCLUDE_DIR}
    ${GLOBAL_GGML_INCLUDE_DIR}
)

add_dependencies(${HOST_APP_NAME} GenerateEDL)

message(STATUS "Configuring Host Application (using OCALL strategy): ${HOST_APP_NAME}")
message(STATUS "  Host sources added: ${CMAKE_CURRENT_SOURCE_DIR}/host.cpp, ${EDL_UNTRUSTED_C_PATH}")

# --- Convenience 'run' Target ---
set(ENCLAVE_TARGET_NAME enclave_prod)
set(ENCLAVE_BUILD_OUTPUT_DIR "${CMAKE_BINARY_DIR}/enclave")
set(SIGNED_ENCLAVE_FULL_PATH "${ENCLAVE_BUILD_OUTPUT_DIR}/${ENCLAVE_TARGET_NAME}.signed.so")

set(MODEL_PATH_FOR_RUN_TARGET "${CMAKE_BINARY_DIR}/model/bert.bin")

message(STATUS "[Host CMake] Model destination path for run target: ${MODEL_PATH_FOR_RUN_TARGET}")

# The paths to the tokenizer script and its data directory, relative to the root of the project.
set(TOKENIZER_SCRIPT_PATH ${CMAKE_SOURCE_DIR}/tokenize_script.py)
set(TOKENIZER_DATA_DIR ${CMAKE_SOURCE_DIR}/tokenizer)

# 2. Updated the 'run' target to a two-step process.
add_custom_target(run
    # MODIFIED: The command now pipes the content of 'tokens_ready.txt'
    # into the host application, which now expects input from stdin.
    # The user is responsible for creating tokens_ready.txt.
    COMMAND bash -c "cat tokens_ready.txt | ${CMAKE_COMMAND} -E env 'LD_LIBRARY_PATH=$<TARGET_FILE_DIR:bert>:$<TARGET_FILE_DIR:ggml>' ./${HOST_APP_NAME} ${MODEL_PATH_FOR_RUN_TARGET} ${SIGNED_ENCLAVE_FULL_PATH} --use-stdin"
    DEPENDS ${HOST_APP_NAME} ${ENCLAVE_TARGET_NAME}_signed
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} # Execute from build/host/
    COMMENT "Running ML PoC with tokenized input from file. Host: ./${HOST_APP_NAME}"
    VERBATIM
)

add_custom_target(run_simulate
    COMMAND bash -c "cat tokens_ready.txt | ${CMAKE_COMMAND} -E env 'LD_LIBRARY_PATH=$<TARGET_FILE_DIR:bert>:$<TARGET_FILE_DIR:ggml>' ./${HOST_APP_NAME} ${MODEL_PATH_FOR_RUN_TARGET} ${SIGNED_ENCLAVE_FULL_PATH} --use-stdin --simulate"
    DEPENDS ${HOST_APP_NAME} ${ENCLAVE_TARGET_NAME}_signed
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} # Execute from build/host/
    COMMENT "Running ML PoC in SIMULATION mode with tokenized input from file."
    VERBATIM
)

message(STATUS "  To run: 'make run' or 'make run_simulate' (after successful build).")
