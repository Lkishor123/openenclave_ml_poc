# Stage 1: Build the C++ Host and Enclave
FROM ubuntu:20.04 AS builder

# Install build dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    wget \
    openssl \
    libssl-dev \
    pkg-config \
    ninja-build

# Install runtime dependencies without sudo
RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main' | tee /etc/apt/sources.list.d/intel-sgx.list \
    && wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -

RUN echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-11 main" | tee /etc/apt/sources.list.d/llvm-toolchain-focal-11.list \
    && wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -

RUN echo "deb [arch=amd64] https://packages.microsoft.com/ubuntu/20.04/prod focal main" | tee /etc/apt/sources.list.d/msprod.list \
    && wget -qO - https://packages.microsoft.com/keys/microsoft.asc | apt-key add -


    # Now install the actual packages
RUN apt-get update && apt-get install -y \
    dkms \
    clang-11 \
    libssl-dev \
    gdb \
    libsgx-enclave-common \
    libsgx-quote-ex \
    libprotobuf17 \
    libsgx-dcap-ql \
    libsgx-dcap-ql-dev \
    az-dcap-client \
    && rm -rf /var/lib/apt/lists/*

# Install Open Enclave SDK
# RUN git clone -b v0.19.0 --recursive --depth 1 https://github.com/openenclave/openenclave && \
#     cd openenclave && \
#     mkdir build && cd build && \
#     cmake -GNinja -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=/opt/openenclave -DHAS_QUOTE_PROVIDER=OFF -DENABLE_REFMAN=OFF .. && \
#     cmake --build . --target install && \
#     echo "source /opt/openenclave/share/openenclave/openenclaverc" >> ~/.bashrc && \
#     source /opt/openenclave/share/openenclave/openenclaverc

RUN wget https://github.com/openenclave/openenclave/releases/download/v0.19.0/Ubuntu_2004_open-enclave_0.19.0_amd64.deb && \
    apt-get install -y ./Ubuntu_2004_open-enclave_0.19.0_amd64.deb
# Activate OE - Note: This only affects this RUN command, not subsequent ones.
# The sourcing is correctly done in the build step below.

# Install ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz && \
    tar -zxvf onnxruntime-linux-x64-1.10.0.tgz -C /opt && \
    mv /opt/onnxruntime-linux-x64-1.10.0 /opt/onnxruntime

# Copy source code
COPY . /app
WORKDIR /app

# Build the C++ application
# Note: The WORKDIR is /app, so we cd into build from there.
RUN rm -rf build && mkdir build && \
    /opt/openenclave/bin/oeedger8r --trusted common/enclave.edl --trusted-dir build/edl_generated --search-path /opt/openenclave/include && \
    /opt/openenclave/bin/oeedger8r --untrusted common/enclave.edl --untrusted-dir build/edl_generated --search-path /opt/openenclave/include && \
    cd build && \
    . /opt/openenclave/share/openenclave/openenclaverc && \
    cmake .. -DONNXRUNTIME_ROOT_DIR=/opt/onnxruntime && \
    make

# Stage 2: Build the Go Backend
FROM golang:1.18-alpine AS go-builder

WORKDIR /app
# Copy go.mod and go.sum first to leverage Docker layer caching
COPY backend/go.mod ./
COPY backend/main.go .
RUN go mod tidy
RUN go build -o /main .

# Stage 3: Final Production Image
FROM ubuntu:20.04

# Install sudo and wget first, as they are used in subsequent commands
RUN apt-get update && apt-get install -y sudo wget gnupg

# Install runtime dependencies without sudo
RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main' | tee /etc/apt/sources.list.d/intel-sgx.list \
    && wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -

RUN echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-11 main" | tee /etc/apt/sources.list.d/llvm-toolchain-focal-11.list \
    && wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -

RUN echo "deb [arch=amd64] https://packages.microsoft.com/ubuntu/20.04/prod focal main" | tee /etc/apt/sources.list.d/msprod.list \
    && wget -qO - https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

# Now install the actual packages
RUN apt-get update && apt-get install -y \
    dkms \
    clang-11 \
    libssl-dev \
    gdb \
    libsgx-enclave-common \
    libsgx-quote-ex \
    libprotobuf17 \
    libsgx-dcap-ql \
    libsgx-dcap-ql-dev \
    az-dcap-client \
    open-enclave \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built artifacts from previous stages
COPY --from=builder /app/build/host/ml_host_prod_go ./ml_host_prod_go
COPY --from=builder /app/build/enclave/enclave_prod.signed.so ./enclave/enclave_prod.signed.so
COPY --from=builder /app/model/simple_model.onnx ./model/simple_model.onnx
COPY --from=go-builder /main ./
COPY frontend ./frontend

# Expose the port the Go backend listens on
EXPOSE 8080

# Command to run the Go backend
CMD ["/app/main"]