#!bin/bash
port=$(shuf -i25000-30000 -n1)

# method_name="lmsys/vicuna-7b-v1.5" # slow
# method_name="meta/Llama-2-7b-chat-hf"
# method_name="google/gemma-7b-it"
method_name="tiiuae/falcon-7b-instruct" # slow
# method_name="baichuan-inc/Baichuan2-7B-Chat"  # not support
deepspeed --include=localhost:2 --master_port $port inference/ICL.py  \
    --data_path /home/zcwang/data/data/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name FOMC,C-STANCE,NumGLUE-cm,NumGLUE-ds,ScienceQA,Py150,20Minuten \
    --model_name_or_path /home/zcwang/data/model/$method_name \
    --inference_batch 1 \
    --max_prompt_len 3584 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --demonstrations_num 6 \
    --inference_output_path /home/zcwang/TRACE/outputs/ICL/$method_name > /home/zcwang/TRACE/outputs/ICL/$method_name/infer.log 2>&1 &

exit 0


# for slurm, single gpu
srun --partition=xai --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /mnt/petrelfs/wangxiao/miniconda3/envs/cl/bin/deepspeed --master_port 51417 inference/ICL.py  \
    --data_path /mnt/petrelfs/wangxiao/DATA/LLM-CL-Benchmark/LLM-CL-Benchmark_5000 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/llama2HF/7B-Chat \
    --inference_batch 4 \
    --max_prompt_len 3584 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --demonstrations_num 6 \
    --inference_output_path /mnt/petrelfs/wangxiao/LLM-Continual-Learning/CKPT/ICL/llama2-7b > llama2_7b_ICL_infer.log 2>&1 &




srun --partition=xai --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /mnt/petrelfs/wangxiao/miniconda3/envs/cl/bin/deepspeed --master_port 51408 inference/ICL.py  \
    --data_path /mnt/petrelfs/wangxiao/DATA/LLM-CL-Benchmark/LLM-CL-Benchmark_5000 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/vicuna-7b-v1.5 \
    --inference_batch 4 \
    --max_prompt_len 3584 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --demonstrations_num 6 \
    --inference_output_path /mnt/petrelfs/wangxiao/LLM-Continual-Learning/CKPT/ICL/vicuna-7b > vicuna-7b_ICL_infer.log 2>&1 &

