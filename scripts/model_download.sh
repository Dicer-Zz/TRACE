
model_name=$1
export HF_ENDPOINT=https://hf-mirror.com

cache_dir=/home/zcwang/data/.cache/huggingface/hub
local_dir=/home/zcwang/data/model/$model_name

echo "we are downloading $model_name to $local_dir (cache: $cache_dir)"

huggingface-cli download \
    --resume-download $model_name \
    --cache-dir $cache_dir \
    --local-dir $local_dir > /dev/null 2>&1 &
