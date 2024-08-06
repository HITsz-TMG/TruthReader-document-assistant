module load cuda/12.1 


export base_model=/data/hxs/Checkpoints/Mixtral_13B_Chat
export lora_model=/data/hxs/Output/DocHelper/Mitrixal/Mixtral_13B_Chat_reader_0329/checkpoint-111
export model_name=/vllm/EMPTY
export max_model_len=4096

export device=2
export port=8002


bash $(dirname "$0")/run_vllm.sh




# model_path=/data/hxs/Output/DocHelper/Mitrixal/Mixtral_13B_Chat_reader_0329_checkpoint-165_full
# model=/vllm/EMPTY
# max_model_len=5000

# device=2
# port=8002

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
