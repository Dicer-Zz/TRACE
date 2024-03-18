#!bin/bash
port=$(shuf -i25000-30000 -n1)

method=None
# method=O-LoRA

model=Qwen/Qwen1.5-1.8B-Chat
# model="google/gemma-2b-it"

deepspeed --include=localhost:2 --master_port $port inference/infer_3H.py \
   --data_path /home/zcwang/TRACE/inference/HHH \
   --inference_tasks helpful,harmless \
   --model_name_or_path /home/zcwang/data/model/$model \
   --inference_batch 2 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --CL_method $method \
   --target_modules "q_proj,v_proj" \
   --inference_output_path /home/zcwang/TRACE/outputs/3H/$model > /home/zcwang/TRACE/outputs/3H/$model/infer.log 2>&1 &
