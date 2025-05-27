# ~/openenclave_ml_poc/helloworld/CMakeLists.txt

cmake_minimum_required(VERSION 3.16)
project(HelloWorldSample C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11) # As per your original helloworld sample

# Use find_package as per the Open Enclave SDK documentation
# This should make targets like openenclave::oeedger8r available.
find_package(OpenEnclave CONFIG REQUIRED)
message(STATUS "Found OpenEnclave SDK version: ${OpenEnclave_VERSION}")
message(STATUS "  OpenEnclave CMake module directory: ${OpenEnclave_DIR}")

# --- EDL File Processing ---
# Define the location of your EDL file.
# Assuming it's in a 'common' subdirectory relative to this CMakeLists.txt
set(EDL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/common/helloworld.edl) # Ensure this path is correct

# Define where the generated EDL files will be placed (within the build directory)
set(EDL_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/edl_generated)

# Ensure the output directory for EDL files exists before edger8r runs
add_custom_command(OUTPUT ${EDL_GENERATED_DIR}/.dir_stamp # Dummy file to represent directory creation
    COMMAND ${CMAKE_COMMAND} -E make_directory ${EDL_GENERATED_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${EDL_GENERATED_DIR}/.dir_stamp
    COMMENT "Creating EDL generated files directory: ${EDL_GENERATED_DIR}"
    VERBATIM)

# Define the expected output files from oeedger8r for the trusted part
set(EDL_TRUSTED_C ${EDL_GENERATED_DIR}/helloworld_t.c)
set(EDL_TRUSTED_H ${EDL_GENERATED_DIR}/helloworld_t.h)
# oeedger8r will also generate helloworld_args.h if needed.

# Custom command to run oeedger8r for the trusted part
add_custom_command(
    OUTPUT ${EDL_TRUSTED_C} ${EDL_TRUSTED_H} # Add helloworld_args.h here if your trusted code needs it
    COMMAND openenclave::oeedger8r # Use the imported target
            --trusted ${EDL_FILE}
            --trusted-dir ${EDL_GENERATED_DIR}
            # Example: --search-path /opt/openenclave/include # If needed and not picked up automatically
    DEPENDS ${EDL_FILE} ${EDL_GENERATED_DIR}/.dir_stamp # Depends on EDL file and directory creation
    COMMENT "Running oeedger8r (trusted) on ${EDL_FILE}"
    VERBATIM)

# Define the expected output files from oeedger8r for the untrusted part
set(EDL_UNTRUSTED_C ${EDL_GENERATED_DIR}/helloworld_u.c)
set(EDL_UNTRUSTED_H ${EDL_GENERATED_DIR}/helloworld_u.h)
set(EDL_UNTRUSTED_ARGS_H ${EDL_GENERATED_DIR}/helloworld_args.h) # Common output

# Custom command to run oeedger8r for the untrusted part
add_custom_command(
    OUTPUT ${EDL_UNTRUSTED_C} ${EDL_UNTRUSTED_H} ${EDL_UNTRUSTED_ARGS_H}
    COMMAND openenclave::oeedger8r # Use the imported target
            --untrusted ${EDL_FILE}
            --untrusted-dir ${EDL_GENERATED_DIR}
            # Example: --search-path /opt/openenclave/include # If needed
    DEPENDS ${EDL_FILE} ${EDL_GENERATED_DIR}/.dir_stamp # Depends on EDL file and directory creation
    COMMENT "Running oeedger8r (untrusted) on ${EDL_FILE}"
    VERBATIM)

# Add a custom target for EDL generation.
# This target ensures that the EDL files are generated before targets that depend on them are built.
add_custom_target(GenerateEDL ALL # 'ALL' makes it a default target
    DEPENDS ${EDL_TRUSTED_C} ${EDL_TRUSTED_H} ${EDL_UNTRUSTED_C} ${EDL_UNTRUSTED_H} ${EDL_UNTRUSTED_ARGS_H}
)

# Include directory for generated EDL headers for all targets in this project
# This allows #include "helloworld_t.h" or #include "helloworld_u.h" to work.
include_directories(${EDL_GENERATED_DIR})

# Add subdirectories for host and enclave.
# Their CMakeLists.txt files will define the actual executables.
add_subdirectory(enclave)
add_subdirectory(host)

message(STATUS "Root helloworld CMake configuration complete.")
