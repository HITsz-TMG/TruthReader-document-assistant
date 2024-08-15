module load cuda/12.1 


export base_model='your/model/path'     # change to your path
export lora_model=''    # leave it empty if you don't have another lora modular
export model_name=/vllm/EMPTY
export max_model_len=4096

export device=2
export port=8002


bash $(dirname "$0")/run_vllm.sh



