
if [ -z "$lora_model" ]; then
    docker run --rm --runtime nvidia --gpus "device=${device}" \
        -v ${base_model}:${model_name} \
        -p ${port}:${port} \
        --ipc=host \
        vllm/vllm-openai:latest \
        --model ${model_name} \
        --port ${port} \
        --tensor-parallel-size 1 \
        --max-model-len ${max_model_len} \
        --gpu-memory-utilization 0.98 \
        --disable-log-stats \
        --trust-remote-code \
        --enforce-eager
else
    docker run --rm --runtime nvidia --gpus "device=${device}" \
        -v ${base_model}:/root/base_model  \
        -v ${lora_model}:/root/lora_model \
        -p ${port}:${port} \
        --ipc=host \
        vllm/vllm-openai:latest \
        --model /root/base_model \
        --enable-lora \
        --lora-modules ${model_name}=/root/lora_model \
        --port ${port} \
        --tensor-parallel-size 1 \
        --max-model-len ${max_model_len} \
        --gpu-memory-utilization 0.98 \
        --disable-log-stats \
        --trust-remote-code \
        --enforce-eager
fi

