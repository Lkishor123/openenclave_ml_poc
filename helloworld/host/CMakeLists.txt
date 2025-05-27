# ~/openenclave_ml_poc/helloworld/host/CMakeLists.txt

cmake_minimum_required(VERSION 3.16)

set(HOST_APP_NAME helloworld_host)

# Create the executable target with source files that exist at configuration time.
add_executable(${HOST_APP_NAME} host.c)

# Add the generated untrusted C file to the target's sources.
# The ${EDL_UNTRUSTED_C} variable is defined in the parent CMakeLists.txt.
# The GENERATED property set on this file in the parent scope is important.
target_sources(${HOST_APP_NAME} PRIVATE ${EDL_UNTRUSTED_C})

# Link the host executable against the Open Enclave host library.
target_link_libraries(${HOST_APP_NAME} PRIVATE openenclave::oehost)

# Ensure that the EDL generation target (GenerateEDL from root CMakeLists.txt)
# is completed before attempting to build this host application.
add_dependencies(${HOST_APP_NAME} GenerateEDL)

message(STATUS "Configuring Host Application (using add_executable and target_sources): ${HOST_APP_NAME}")
message(STATUS "  Host base source: host.c")
message(STATUS "  Host generated source: ${EDL_UNTRUSTED_C}")
message(STATUS "  Host executable: ${HOST_APP_NAME}")

# --- Convenience 'run' Targets ---
set(ENCLAVE_BUILD_OUTPUT_DIR "${CMAKE_BINARY_DIR}/enclave")
# ENCLAVE_NAME should be consistent with the one set in enclave/CMakeLists.txt
set(SIGNED_ENCLAVE_FULL_PATH "${ENCLAVE_BUILD_OUTPUT_DIR}/helloworld_enclave.signed.so")

add_custom_target(run
    COMMAND ./${HOST_APP_NAME} ${SIGNED_ENCLAVE_FULL_PATH}
    DEPENDS ${HOST_APP_NAME} helloworld_enclave_signed # Depends on the target that creates the signed enclave
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} # Typically build/host/
    COMMENT "Running helloworld sample with signed enclave."
)

add_custom_target(run_simulate
    COMMAND ./${HOST_APP_NAME} ${SIGNED_ENCLAVE_FULL_PATH} --simulate
    DEPENDS ${HOST_APP_NAME} helloworld_enclave_signed
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running helloworld sample in SIMULATION mode with signed enclave."
)

message(STATUS "  To run: 'make run' or 'make run_simulate' (after successful build).")
message(STATUS "  Run targets expect signed enclave at: ${SIGNED_ENCLAVE_FULL_PATH}")
