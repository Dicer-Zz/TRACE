#!bin/bash
port=$(shuf -i25000-30000 -n1)

# method_name="lmsys/vicuna-7b-v1.5" # slow
# method_name="meta/Llama-2-7b-chat-hf"
# method_name="google/gemma-7b-it"
# method_name="tiiuae/falcon-7b-instruct" # slow
method_name=Qwen/Qwen1.5-0.5B-Chat
# method_name="baichuan-inc/Baichuan2-7B-Chat"  # not support

deepspeed --include=localhost:0 --master_port $port inference/FSICL.py  \
    --data_path /home/zcwang/data/data/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name ScienceQA \
    --model_name_or_path /home/zcwang/data/model/$method_name \
    --inference_batch 4 \
    --max_prompt_len 3584 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --temperature 0.7 \
    --repetition_penalty 0.5 \
    --shots 1 \
    --cot \
    --self_consistency \
    --paths 2 \
    --inference_output_path /home/zcwang/TRACE/outputs/FSICL/$method_name > /home/zcwang/TRACE/outputs/FSICL/$method_name/infer.log 2>&1 &
