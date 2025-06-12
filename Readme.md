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

cmake .. -DGGML_ROOT_DIR=/opt/ggml -DCMAKE_BUILD_TYPE=Debug

make
make run


## Docker Build:
docker build -t confidential-ml-app .
docker run --rm -p 8080:8080 --device /dev/sgx_enclave confidential-ml-app

## For Simulation Mode:
# This requires modifying backend/main.go to pass the --simulate flag
docker run --rm -p 8080:8080 confidential-ml-app


pip install optimum[exporters] transformers
