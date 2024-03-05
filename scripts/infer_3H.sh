#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:6 --master_port $port inference/infer_3H.py \
   --data_path /home/zcwang/TRACE/inference/HHH \
   --inference_tasks helpful,harmless \
   --model_name_or_path /home/zcwang/data/model/google/gemma-2b-it \
   --inference_batch 2 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --inference_output_path /home/zcwang/TRACE/outputs/3H > /home/zcwang/TRACE/outputs/3H/infer_gemma2b_it.log 2>&1 &
