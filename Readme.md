## Openenclave_ML_POC

openssl genrsa -out enclave/enclave_private.pem -3 3072


# Generate trusted code
oeedger8r --trusted common/enclave.edl --trusted-dir build/edl_generated --search-path /opt/openenclave/include

# Generate untrusted code
oeedger8r --untrusted common/enclave.edl --untrusted-dir build/edl_generated --search-path /opt/openenclave/include

# Build
cd openenclave_ml_poc/
mkdir build
cd build/

cmake .. -DONNXRUNTIME_ROOT_DIR=/opt/onnxruntime -DCMAKE_BUILD_TYPE=Debug

make
make run