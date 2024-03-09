#!bin/bash

port=$(shuf -i25000-30000 -n1)
method=EPI
model="google/gemma-2b-it"

deepspeed --include=localhost:0 --master_port $port inference/infer_single.py \
   --data_path /home/zcwang/data/data/TRACE-Benchmark/LLM-CL-Benchmark_500 \
   --inference_tasks C-STANCE,FOMC,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
   --model_name_or_path /home/zcwang/data/model/$model \
   --inference_model_path /home/zcwang/TRACE/outputs/cl/$method/$model \
   --inference_batch 8 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --CL_method $method \
   --inference_output_path /home/zcwang/TRACE/outputs/cl/$method/$model/predictions > /home/zcwang/TRACE/outputs/cl/$method/$model/infer.log 2>&1 &