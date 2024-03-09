#!/bin/bash

port=$(shuf -i25000-30000 -n1)
method=EPI
# model=Qwen/Qwen1.5-1.8B-Chat
model=google/gemma-2b-it

deepspeed --include=localhost:0,1,2,3 --master_port $port training/main.py \
   --data_path /home/zcwang/data/data/TRACE-Benchmark/LLM-CL-Benchmark_500 \
   --dataset_name C-STANCE,FOMC,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
   --model_name_or_path /home/zcwang/data/model/$model \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 16 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.0 \
   --num_train_epochs 1,1,1,1,1,1,1 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --half \
   --target_modules "q_proj,v_proj" \
   --print_loss \
   --CL_method $method \
   --output_dir /home/zcwang/TRACE/outputs/cl/$method/$model > /home/zcwang/TRACE/outputs/cl/$method/$model/train.log 2>&1 &


# 5,3,5,3,5,5,7 \