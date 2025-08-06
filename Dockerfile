# Stage 1: Build the C++ Host and Enclave using the local source and CMake configuration.
FROM ubuntu:20.04 AS builder

# Install build dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    openssl \
    libssl-dev \
    pkg-config \
    wget

# Install Intel SGX and Open Enclave SDK dependencies
RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main' | tee /etc/apt/sources.list.d/intel-sgx.list \
    && wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add - \
    && echo "deb [arch=amd64] https://packages.microsoft.com/ubuntu/20.04/prod focal main" | tee /etc/apt/sources.list.d/msprod.list \
    && wget -qO - https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

ENV DEBIAN_FRONTEND=noninteractive

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
    libsgx-dcap-ql-dev

RUN apt-get install -y az-dcap-client \
    && rm -rf /var/lib/apt/lists/*

# Install Open Enclave SDK from .deb package
RUN wget https://github.com/openenclave/openenclave/releases/download/v0.19.0/Ubuntu_2004_open-enclave_0.19.0_amd64.deb && \
    apt-get update && \
    apt-get install -y ./Ubuntu_2004_open-enclave_0.19.0_amd64.deb && \
    rm ./Ubuntu_2004_open-enclave_0.19.0_amd64.deb

# Copy the entire project context into the container.
# This includes the pre-downloaded external/bert.cpp and model/bert.bin
COPY . /app
WORKDIR /app

# Build the C++ application using the root CMakeLists.txt
# This now mirrors the successful local build process.
RUN rm -rf build && mkdir build && \
    /opt/openenclave/bin/oeedger8r --trusted common/enclave.edl --trusted-dir build/edl_generated --search-path /opt/openenclave/include && \
    /opt/openenclave/bin/oeedger8r --untrusted common/enclave.edl --untrusted-dir build/edl_generated --search-path /opt/openenclave/include && \
    cd build && \
    . /opt/openenclave/share/openenclave/openenclaverc && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make

# Stage 2: Build the Go Backend
FROM golang:1.21-alpine AS go-builder

WORKDIR /app
# Copy only the necessary files for the Go build
COPY backend/go.mod ./
COPY backend/main.go .
RUN go mod tidy
# Build the Go binary statically to avoid C dependencies in the final image.
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o /main .

# Stage 3: Final Production Image
FROM ubuntu:20.04

# Install only the necessary RUNTIME dependencies
RUN apt-get update && apt-get install -y sudo wget gnupg

# Install runtime dependencies without sudo
RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main' | tee /etc/apt/sources.list.d/intel-sgx.list \
    && wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -

RUN echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-11 main" | tee /etc/apt/sources.list.d/llvm-toolchain-focal-11.list \
    && wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -

RUN echo "deb [arch=amd64] https://packages.microsoft.com/ubuntu/20.04/prod focal main" | tee /etc/apt/sources.list.d/msprod.list \
    && wget -qO - https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

 # Set the DEBIAN_FRONTEND variable to noninteractive to bypass prompts
ENV DEBIAN_FRONTEND=noninteractive

# Now install the actual packages
RUN apt-get update && apt-get install -y \
    dkms \
    ca-certificates \
    clang-11 \
    libssl-dev \
    gdb \
    libsgx-enclave-common \
    libsgx-quote-ex \
    libprotobuf17 \
    libsgx-dcap-ql \
    libsgx-dcap-ql-dev \
    az-dcap-client \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/openenclave/openenclave/releases/download/v0.19.0/Ubuntu_2004_open-enclave_0.19.0_amd64.deb && \
    apt-get install -y ./Ubuntu_2004_open-enclave_0.19.0_amd64.deb

# Install the Python transformers library needed by the tokenizer script
RUN pip3 install --no-cache-dir transformers

WORKDIR /app

# Copy necessary artifacts from the builder stage
COPY --from=builder /app/build/host/ml_host_prod_go ./ml_host_prod_go
COPY --from=builder /app/build/enclave/enclave_prod.signed.so ./enclave/enclave_prod.signed.so
# Copy the shared libraries required by the host application
COPY --from=builder /app/build/external_bertcpp_build/libbert.so /usr/local/lib/
COPY --from=builder /app/build/external_bertcpp_build/ggml/src/libggml.so /usr/local/lib/

# Copy artifacts from the Go builder stage
COPY --from=go-builder /main ./main

# Copy application assets from the original build context
COPY frontend ./frontend
COPY model/bert.bin ./model/bert.bin
COPY tokenizer ./tokenizer
COPY tokenize_script.py .

# Update the linker cache to find the newly added shared libraries
RUN ldconfig

# Expose the port the Go backend listens on
EXPOSE 8080

# Command to run the Go backend
CMD ["/app/main"]
