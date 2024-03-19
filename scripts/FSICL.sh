#!bin/bash
port=$(shuf -i25000-30000 -n1)

# method_name="lmsys/vicuna-7b-v1.5" # slow
# method_name="meta/Llama-2-7b-chat-hf"
# method_name="google/gemma-2b-it"
# method_name="tiiuae/falcon-7b-instruct" # slow
method_name=Qwen/Qwen1.5-1.8B-Chat
# method_name="baichuan-inc/Baichuan2-7B-Chat"  # not support

deepspeed --include=localhost:2 --master_port $port inference/FSICL.py  \
    --data_path /home/zcwang/data/data/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name 20Minuten,Py150 \
    --model_name_or_path /home/zcwang/data/model/$method_name \
    --inference_batch 1 \
    --max_prompt_len 3584 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --temperature 0.7 \
    --shots 6 \
    --cot \
    --inference_output_path /home/zcwang/TRACE/outputs/FSICL/$method_name/CoT > /home/zcwang/TRACE/outputs/FSICL/$method_name/CoT/infer.log 2>&1 &
