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

# MODIFIED: Link against both bert and ggml. This ensures that the host
# has access to both libraries and, crucially, their include paths.
target_link_libraries(${HOST_APP_NAME} PRIVATE
    openenclave::oehost
    bert
    ggml
    stdc++fs
)

# MODIFIED: Explicitly add the include directories for bert.cpp and its ggml submodule.
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

# This property is not available at configure time. We get the directory from the lib name.
# This is less ideal, but works for the run target's environment variable.
get_target_property(BERT_LIB_LOCATION bert LOCATION)
get_filename_component(HOST_LIB_DIR ${BERT_LIB_LOCATION} DIRECTORY)

message(STATUS "[Host CMake] GGML/Bert Library Directory for LD_LIBRARY_PATH: ${HOST_LIB_DIR}")
message(STATUS "[Host CMake] Model destination path for run target: ${MODEL_PATH_FOR_RUN_TARGET}")

# 2. Updated the 'run' target to work with the new stdin-based host
# This now pipes a sample input string to the executable for testing.
# NOTE: The GGML model must be downloaded via scripts/download_deps.sh
# before attempting to run this target.
add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${HOST_LIB_DIR}" bash -c "if [ ! -f \"${MODEL_PATH_FOR_RUN_TARGET}\" ]; then echo 'Model file missing at ${MODEL_PATH_FOR_RUN_TARGET}. Run scripts/download_deps.sh first.'; exit 1; fi; echo '3.14,-2.71' | ./${HOST_APP_NAME} ${MODEL_PATH_FOR_RUN_TARGET} ${SIGNED_ENCLAVE_FULL_PATH} --use-stdin"
    DEPENDS ${HOST_APP_NAME} ${ENCLAVE_TARGET_NAME}_signed
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} # Execute from build/host/
    COMMENT "Running ML PoC. Host: ./${HOST_APP_NAME}, Model: ${MODEL_PATH_FOR_RUN_TARGET}, Enclave: ${SIGNED_ENCLAVE_FULL_PATH}"
    VERBATIM
)

add_custom_target(run_simulate
    COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${HOST_LIB_DIR}" bash -c "if [ ! -f \"${MODEL_PATH_FOR_RUN_TARGET}\" ]; then echo 'Model file missing at ${MODEL_PATH_FOR_RUN_TARGET}. Run scripts/download_deps.sh first.'; exit 1; fi; echo '3.14,-2.71' | ./${HOST_APP_NAME} ${MODEL_PATH_FOR_RUN_TARGET} ${SIGNED_ENCLAVE_FULL_PATH} --use-stdin --simulate"
    DEPENDS ${HOST_APP_NAME} ${ENCLAVE_TARGET_NAME}_signed
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} # Execute from build/host/
    COMMENT "Running ML PoC in SIMULATION mode."
    VERBATIM
)

message(STATUS "  To run: 'make run' or 'make run_simulate' (after successful build).")
