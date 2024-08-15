#!/bin/bash


export GRADIO_ANALYTICS_ENABLED=False


cd $(dirname "$0")/../src


export embed_model_path="your_local_embedding_model_path"
export chat_model_path="your_local_mixtral_tokenizer_path;your_local_qwen_tokenizer_path"
export chat_model_server="your_generator_address_it_can_be_http://0.0.0.0"
export ocr_model_path="your_local_ocr_model_path"

export log_dir="your_local_folder_to_store_logging"
export GRADIO_TEMP_DIR="your_local_folder_to_store_uploaded_documents"

python3 main.py \
  --embed_model_path ${embed_model_path} \
  --chat_model_path ${chat_model_path} \
  --chat_model_server ${chat_model_server} \
  --ocr_model_path ${ocr_model_path} \
  --log_dir ${log_dir} \
  --tmp_folder ${GRADIO_TEMP_DIR} \
  --port 8080 


