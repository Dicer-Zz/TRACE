#!bin/bash
port=$(shuf -i25000-30000 -n1)

method=replay
# model="google/gemma-2b-it"
model="Qwen/Qwen1.5-1.8B-Chat"

deepspeed --include=localhost:0 --master_port $port inference/infer_single.py \
   --data_path /home/zcwang/data/data/TRACE-Benchmark/LLM-CL-Benchmark_500 \
   --inference_tasks C-STANCE,FOMC,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
   --model_name_or_path /home/zcwang/data/model/$model \
   --inference_model_path /home/zcwang/TRACE/outputs/cl/replay/$model \
   --inference_batch 4 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --CL_method lora \
   --target_modules "q_proj,v_proj" \
   --inference_output_path /home/zcwang/TRACE/outputs/cl/replay/$model/predictions > /home/zcwang/TRACE/outputs/cl/replay/$model/infer.log 2>&1 &
