#!/bin/bash
echo ":: Starting inference endpoint"
cd LLaMA-Factory
CUDA_VISIBLE_DEVICES=0 API_PORT=8000 llamafactory-cli api \
    --model_name_or_path unsloth/llama-3-8b-Instruct-bnb-4bit \
    --adapter_name_or_path saves/LLaMA3-8B/sft4  \
    --template llama3 \
    --quantization_bit 4 \
    --finetuning_type lora \
    > ../inference.log &
endpoint_pid="$!"
while ! nc -z 0.0.0.0 8000 > /dev/null 2>&1; do
    sleep 1
done
echo ":: Inference for test data started."
cd ..
python generate_submission.py
echo ":: Closing endpoint"
kill -9 $endpoint_pid
while nc -z 0.0.0.0 8000 > /dev/null 2>&1; do
    sleep 1
done
echo ":: Endpoint closed."