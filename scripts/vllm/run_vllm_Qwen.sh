module load cuda/12.1 


export base_model='your/model/path'     # change to your path
export lora_model=''    # leave it empty if you don't have another lora modular
export model_name=/vllm/EMPTY
export max_model_len=5000

export device=1
export port=8001


bash $(dirname "$0")/run_vllm.sh


