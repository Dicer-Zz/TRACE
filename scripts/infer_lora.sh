#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:7 --master_port $port inference/infer_single.py \
   --data_path /home/zcwang/data/data/TRACE-Benchmark/LLM-CL-Benchmark_500 \
   --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
   --model_name_or_path /home/zcwang/data/model/google/gemma-2b-it \
   --inference_model_path /home/zcwang/TRACE/outputs/cl/O-LoRA \
   --inference_batch 1 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --CL_method O-LoRA \
   --inference_output_path /home/zcwang/TRACE/outputs/cl/O-LoRA/predictions > /home/zcwang/TRACE/outputs/cl/O-LoRA/infer.log 2>&1 &
   
