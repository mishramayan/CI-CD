#!/bin/bash
# This script is meant to be run inside the docker container

output_dir="output"
model_name="hello_world_int8"

# Create output directory
mkdir -p "$output_dir"

# Function to check errors and log to a file
check_and_log() {
    local log="$1"
    local log_filename="$2"
    local action_name="$3"

    if [ $? -ne 0 ]; then
        echo "$log" | tee $output_dir/${log_filename}_error.log
        echo "An error occurred during $action_name. Please check the error log for details."
        exit 1
    else
        echo "$log" | tee $output_dir/${log_filename}.log
    fi
}

echo "Running vela..."
vela_logs=$(vela --accelerator-config ethos-u55-64 --config himax_vela.ini \
    --system-config My_Sys_Cfg --memory-mode My_Mem_Mode_Parent \
    --verbose-all \
    --output-dir "$output_dir" \
    --enable-debug-db \
    --timing \
    "${model_name}.tflite" 2>&1)

check_and_log "$vela_logs" "vela" "vela conversion."

echo "Running xxd..."
xxd_log=$(xxd -i "$output_dir/${model_name}_vela.tflite" |
    sed -e "s/${output_dir}_//g" \
        -e "s/^unsigned/extern const unsigned/g" \
        -e 's/\[\]/[] __attribute__((section(".tflite_model"), aligned(16)))/' \
        >"$output_dir/yolo_coco_vela.cc" 2>&1)

check_and_log "$xxd_log" "xxd" "c file generation."
