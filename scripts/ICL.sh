#!bin/bash
port=$(shuf -i25000-30000 -n1)

# method_name="lmsys/vicuna-7b-v1.5" # slow
# method_name="meta/Llama-2-7b-chat-hf"
# method_name="google/gemma-7b-it"
# method_name="tiiuae/falcon-7b-instruct" # slow
method_name=Qwen/Qwen1.5-1.8B-Chat
# method_name="baichuan-inc/Baichuan2-7B-Chat"  # not support
deepspeed --include=localhost:0 --master_port $port inference/ICL.py  \
    --data_path /home/zcwang/data/data/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name Py150,20Minuten \
    --model_name_or_path /home/zcwang/data/model/$method_name \
    --inference_batch 1 \
    --max_prompt_len 3584 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --demonstrations_num 6 \
    --inference_output_path /home/zcwang/TRACE/outputs/ICL/$method_name > /home/zcwang/TRACE/outputs/ICL/$method_name/infer.log 2>&1 &
