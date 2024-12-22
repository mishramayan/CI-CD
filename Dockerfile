# Use a base image with TensorFlow 2.17.0 (CPU version)
FROM tensorflow/tensorflow:2.17.0

ARG HOST_GID=1000
ARG HOST_UID=1000

# Install necessary tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends xxd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install `flatc`
RUN wget -q https://github.com/PINTO0309/onnx2tf/releases/download/1.16.31/flatc.tar.gz \
    && tar -zxvf flatc.tar.gz \
    && chmod +x flatc \
    && mv flatc /usr/local/bin/ \
    && rm flatc.tar.gz

# Install required Python packages
RUN pip install --ignore-installed blinker
RUN pip install --no-cache-dir \
    tensorflow==2.17.0 \
    torch \
    onnx==1.16.1 \
    onnxruntime==1.18.1 \
    onnxsim==0.4.33 \
    simple_onnx_processing_tools \
    onnx2tf \
    psutil==5.9.5 \
    tf-keras~=2.16 \
    matplotlib==3.9.2 \
    ethos-u-vela \
    mlflow==2.18.0

RUN pip install --no-cache-dir \
    onnx_graphsurgeon \
    --index-url https://pypi.ngc.nvidia.com \
    --trusted-host pypi.ngc.nvidia.com
# Notes:
# --no-cache-dir: Do not cache downloaded packages
# --workaround for ssl issue in corporate network

WORKDIR /workspace

CMD ["/bin/bash"]
