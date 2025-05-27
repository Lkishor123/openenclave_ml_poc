# ~/openenclave_ml_poc/helloworld/enclave/CMakeLists.txt

cmake_minimum_required(VERSION 3.16)

set(ENCLAVE_NAME helloworld_enclave)

# 1. Create the executable target initially WITHOUT listing any source files.
#    Line 8 in your error message likely corresponds to this line if it's different.
add_executable(${ENCLAVE_NAME} "") # Ensure this line has NO source files listed.

# 2. Add all source files (existing and generated) to the target using target_sources.
#    The ${EDL_TRUSTED_C} variable is defined in the parent CMakeLists.txt,
#    and its GENERATED property is set there, which is critical.
target_sources(${ENCLAVE_NAME} PRIVATE
    enc.c             # Your existing, non-generated source file
    ${EDL_TRUSTED_C}  # The generated source file from EDL
)

# Link the enclave executable against the required Open Enclave libraries.
target_link_libraries(${ENCLAVE_NAME} PRIVATE openenclave::oeenclave openenclave::oelibc)

# Optional: Set the Open Enclave API version
# target_compile_definitions(${ENCLAVE_NAME} PRIVATE OE_API_VERSION=2)

# Ensure that the EDL generation target (GenerateEDL from root CMakeLists.txt)
# is completed before attempting to build this enclave.
add_dependencies(${ENCLAVE_NAME} GenerateEDL)

message(STATUS "Configuring Enclave (using add_executable \"\" and target_sources): ${ENCLAVE_NAME}")
message(STATUS "  Enclave sources added via target_sources: enc.c, ${EDL_TRUSTED_C}")
message(STATUS "  Unsigned enclave executable: ${ENCLAVE_NAME}")

# --- Enclave Signing ---
# (Ensure helloworld.conf and private.pem are in this directory: helloworld/enclave/)
set(ENCLAVE_CONF_FILE ${CMAKE_CURRENT_SOURCE_DIR}/helloworld.conf)
set(ENCLAVE_PRIVATE_KEY_FILE ${CMAKE_CURRENT_SOURCE_DIR}/private.pem)

# Optional: Command to generate a demo private key if it doesn't exist.
find_package(OpenSSL)
if(OpenSSL_FOUND AND NOT EXISTS ${ENCLAVE_PRIVATE_KEY_FILE})
    add_custom_command(OUTPUT ${ENCLAVE_PRIVATE_KEY_FILE}
        COMMAND ${OpenSSL_EXECUTABLE} genrsa -out ${ENCLAVE_PRIVATE_KEY_FILE} -3 3072
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating self-signed key pair (private.pem) for enclave signing (DEMO ONLY, if key not present)"
        VERBATIM)
    message(STATUS "Attempting to generate demo key: ${ENCLAVE_PRIVATE_KEY_FILE} (if it does not exist)")
else()
    if(NOT OpenSSL_FOUND)
        message(WARNING "OpenSSL not found. Cannot generate demo private key. Ensure ${ENCLAVE_PRIVATE_KEY_FILE} exists.")
    endif()
    if(EXISTS ${ENCLAVE_PRIVATE_KEY_FILE})
         message(STATUS "Using existing private key: ${ENCLAVE_PRIVATE_KEY_FILE}")
    else()
         message(WARNING "Private key ${ENCLAVE_PRIVATE_KEY_FILE} not found and OpenSSL not available to generate it (or key generation command not run).")
    endif()
endif()

set(SIGNED_ENCLAVE_FILE ${CMAKE_CURRENT_BINARY_DIR}/${ENCLAVE_NAME}.signed.so)

add_custom_command(OUTPUT ${SIGNED_ENCLAVE_FILE}
    DEPENDS
        ${ENCLAVE_NAME}                 # Depends on the unsigned enclave executable
        ${ENCLAVE_CONF_FILE}            # Depends on the enclave configuration file
        ${ENCLAVE_PRIVATE_KEY_FILE}     # Depends on the private key
    COMMAND openenclave::oesign sign
            -e $<TARGET_FILE:${ENCLAVE_NAME}>
            -c ${ENCLAVE_CONF_FILE}
            -k ${ENCLAVE_PRIVATE_KEY_FILE}
    COMMENT "Signing enclave ${ENCLAVE_NAME} -> ${SIGNED_ENCLAVE_FILE}"
    VERBATIM)

add_custom_target(${ENCLAVE_NAME}_signed ALL
    DEPENDS ${SIGNED_ENCLAVE_FILE})

message(STATUS "  Signed enclave will be: ${SIGNED_ENCLAVE_FILE}")
message(STATUS "  Ensure ${ENCLAVE_CONF_FILE} and ${ENCLAVE_PRIVATE_KEY_FILE} exist for signing.")
