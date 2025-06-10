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
    pkg-config

# Install Open Enclave SDK
RUN wget https://github.com/openenclave/openenclave/releases/download/v0.17.4/open-enclave-0.17.4-linux-x64.deb && \
    apt-get install -y ./open-enclave-0.17.4-linux-x64.deb
# Activate OE
RUN . /opt/openenclave/share/openenclave/openenclaverc

# Install ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz && \
    tar -zxvf onnxruntime-linux-x64-1.10.0.tgz -C /opt && \
    mv /opt/onnxruntime-linux-x64-1.10.0 /opt/onnxruntime

# Copy source code
COPY . /app
WORKDIR /app

# Build the C++ application
RUN cd build && \
    . /opt/openenclave/share/openenclave/openenclaverc && \
    cmake .. -DONNXRUNTIME_ROOT_DIR=/opt/onnxruntime && \
    make

# Stage 2: Build the Go Backend
FROM golang:1.18-alpine AS go-builder

WORKDIR /app
COPY backend/go.mod ./
RUN go mod download
COPY backend/main.go .
RUN go build -o /main .

# Stage 3: Final Production Image
FROM ubuntu:20.04

# Install only runtime dependencies
RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list \
    && wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | sudo apt-key add -

RUN echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-11 main" | sudo tee /etc/apt/sources.list.d/llvm-toolchain-focal-11.list \
    && wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -

RUN echo "deb [arch=amd64] https://packages.microsoft.com/ubuntu/20.04/prod focal main" | sudo tee /etc/apt/sources.list.d/msprod.list \
    && wget -qO - https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -

RUN apt-get update && apt -y install dkms install clang-11 libssl-dev \
    gdb libsgx-enclave-common libsgx-quote-ex libprotobuf17 libsgx-dcap-ql libsgx-dcap-ql-dev \
    az-dcap-client open-enclave

RUN rm -rf /var/lib/apt/lists/*

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