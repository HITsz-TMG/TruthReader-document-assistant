module load cuda/12.1 


export base_model=/data/hxs/Checkpoints/Qwen1.5-14B-Chat
export lora_model=/data/hxs/Output/DocHelper/Qwen1.5/Qwen1.5-14B-Chat_reader_0315/checkpoint-232
# export lora_model=/data/hxs/Output/DocHelper/Qwen1.5/Qwen1.5-14B-Chat_reader_0329/checkpoint-112
export model_name=/vllm/EMPTY
export max_model_len=5000

export device=1
export port=8001


bash $(dirname "$0")/run_vllm.sh










# model_path=/data/hxs/Output/DocHelper/Qwen1.5/Qwen1.5-14B-Chat_reader_0315_checkpoint-232_full
# model=/vllm/EMPTY
# max_model_len=5000

# device=1
# port=8001

# docker run --rm --runtime nvidia --gpus "device=${device}" \
#     -v ${model_path}:${model} \
#     -p ${port}:${port} \
#     --ipc=host \
#     vllm/vllm-openai:latest \
#     --model ${model} \
#     --port ${port} \
#     --tensor-parallel-size 1 \
#     --max-model-len ${max_model_len} \
#     --gpu-memory-utilization 0.98 \
#     --disable-log-stats \
#     --trust-remote-code \
#     --enforce-eager