#!/bin/bash


export CUDA_VISIBLE_DEVICES=3

export GRADIO_ANALYTICS_ENABLED=False


cd /raid/hxs/python_workspace/Demo/DocHelper/src


export embed_model_path=/raid/hxs/Output/DocHelper/DPR/bge-m3_refgpt_raw_zh_train_0227
export chat_model_path="/raid/hxs/Output/DocHelper/chatbot/LyChee-6B/SFT_eos-161000-filtered_hf/0315/global_step390;/raid/hxs/Output/DocHelper/chatbot/Baichuan/SFT/baichuan2-13b-chat_reader_0315;/raid/hxs/Output/DocHelper/chatbot/Qwen1.5-14B/SFT/Qwen1.5-14B-Chat_reader_0315_checkpoint-232_full;/raid/hxs/Output/DocHelper/chatbot/Mixtral/SFT/Mixtral_13B_Chat_reader_0329_checkpoint-165_full"
export chat_model_server="http://219.223.251.156"
export orc_model_path="/raid/hxs/Checkpoints/huggingface_models/nougat-small-onnx"

export log_dir=/raid/hxs/log_dir/chatbot/log/
export GRADIO_TEMP_DIR=/raid/hxs/log_dir/chatbot/docs/

proxychains4 -f /raid/hxs/proxychains4.conf  python3 main.py \
  --embed_model_path ${embed_model_path} \
  --chat_model_path ${chat_model_path} \
  --chat_model_server ${chat_model_server} \
  --orc_model_path ${orc_model_path} \
  --tmp_folder ${GRADIO_TEMP_DIR} \
  --port 8080 


# rm -rf ${GRADIO_TEMP_DIR}/*
