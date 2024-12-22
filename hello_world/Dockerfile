# Use a base image with TensorFlow 2.17.0 (CPU version)
FROM tensorflow/tensorflow:2.17.0

# Install necessary tools and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python-is-python3 \
    wget \
    && apt-get clean

# Download and install `flatc`
RUN wget https://github.com/PINTO0309/onnx2tf/releases/download/1.16.31/flatc.tar.gz \
    && tar -zxvf flatc.tar.gz \
    && chmod +x flatc \
    && mv flatc /usr/bin/ \
    && rm flatc.tar.gz

# Install additional required libraries
RUN pip install torch onnx onnxruntime

# Upgrade pip and install required Python packages
RUN pip install -U pip \
    && pip install tensorflow==2.17.0 \
    && pip install -U onnx==1.16.1 \
    && python -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
    && pip install -U onnxruntime==1.18.1 \
    && pip install -U onnxsim==0.4.33 \
    && pip install -U simple_onnx_processing_tools \
    && pip install -U onnx2tf \
    && pip install -U protobuf==3.20.3 \
    && pip install -U h5py==3.11.0 \
    && pip install -U psutil==5.9.5 \
    && pip install -U ml_dtypes==0.3.2 \
    && pip install -U tf-keras~=2.16 \
    && pip install flatbuffers>=23.5.26

# Install vela compiler
RUN pip install ethos-u-vela

# Install xxd 
RUN apt install xxd 


# Set the working directory inside the container
WORKDIR /workspace

# copy onnx files and other required files to the container
COPY hello_world_int8.tflite .

COPY himax_vela.ini .


RUN vela --accelerator-config ethos-u55-64 --config himax_vela.ini --system-config My_Sys_Cfg --memory-mode My_Mem_Mode_Parent hello_world_int8.tflite > vela_log.txt  


RUN xxd -i output/hello_world_int8_vela.tflite > hello_world_int8_vela.cc

ENTRYPOINT ["/bin/bash"]



#ToDo: run the onnx2tf command from inside the python. That way, the onnx file can remain on github and the container can be restarted 
# and the python code gets run on restart due to entrypoint clause. In the python code, we can pull the onnx file from github and convert it. 

