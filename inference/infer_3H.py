import sys
sys.dont_write_bytecode = True
import argparse
import os
from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    get_constant_schedule_with_warmup
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import safe_get_full_grad


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


from inference.HHH.HHH_data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten # to be continued
from inference.HHH.data_process import HHH
from datasets import load_dataset

# from training.params import Method2Class, AllDatasetName
from model.Dynamic_network.EPI import EPI

# # add flash attention
# from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
# from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn

# replace_llama_attn_with_flash_attn()
# replace_bloom_attn_with_flash_attn()
# dist.init_process_group(backend='nccl')


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset. A single data path.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--inference_tasks",
        type=list_of_strings,
        default='all',
        help='Datasets to be used.'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    # inference params
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=256,
        help="The maximum answer length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generate temperature params.",
    )
    parser.add_argument(
        "--inference_batch",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    # TODO, add other inference params
    parser.add_argument(
        "--inference_task",
        type=str,
        default=None,
        help="Which task to be infered"
    )
    # lora settings
    parser.add_argument('--target_modules',
                        type=list_of_strings,
                        default=None,
                        help='Target modules for LoRA adapter.')
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")

    # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
    # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # added by wangxiao
    parser.add_argument('--inference_output_path',
                        type=str,
                        default=None,
                        help="Where to store inference results.")
    parser.add_argument('--CL_method',
            default=None,
            help='continual learning method used')
    parser.add_argument(
        "--inference_model_path",
        type=str,
        help=
        "Path to inference model.",
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # local_rank = int(os.getenv('LOCAL_RANK', '0'))
    # 自动获取 word_size
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # Barrier to make sure all process are ready to train
    # torch.distributed.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)

    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    # set evaluation batch size
    # only support bs = 1, cause right padding training logic
    # TODO, modify left pad for training and inference
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=None,
                            )
    # model = model.bfloat16()

    if args.CL_method == "EPI":
        from utils.our_peft import LoraModel, LoraConfig

        lora_config = LoraConfig(
            r=8, lora_alpha=32,
            target_modules=args.target_modules)

        # prepare the model for EPI
        model = LoraModel(model, lora_config, adapter_name="task_neo")

        # ! we have to expand our model before the deepspeed initialization
        # ! because in the training of deepspeed, the model not allow to change the model structure
        task_count = 7
        for task_num in range(0, task_count):
            new_task = f"task_{task_num}"
            model.peft_config[new_task] = lora_config
            model.inject_adapter(model, new_task)

        model_state_dict = torch.load(os.path.join(args.inference_model_path, "pytorch_model.bin"))
        for name, param in model.named_parameters():
            assert name in model_state_dict
            param.data.copy_(model_state_dict[name])

        model = EPI.load_model(model, args.inference_model_path, args)

        model.model.to(device)
    else:
        model.to(device)

    # reference
    # https://github.com/microsoft/DeepSpeed/blob/master/docs/_tutorials/inference-tutorial.md
    # https://huggingface.co/docs/transformers/main_classes/pipelines
    # https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/text-generation/inference-test.py
    # https://discuss.huggingface.co/t/using-text-generation-pipeline-for-llama-2-7b-chat-hf-setting-high-t-doesnt-change-output/48982
    # https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/
    # https://www.deepspeed.ai/tutorials/inference-tutorial/

    # replace_with_kernel_inject = False if "falcon" in args.model_name_or_path.lower() else True
    # replace_with_kernel_inject = False
    # ds_engine = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.bfloat16, checkpoint=None,
    #                                      replace_with_kernel_inject=replace_with_kernel_inject,
    #                                      max_out_tokens=args.max_prompt_len + args.max_ans_len)
    # model = ds_engine.module

    from transformers import GenerationConfig
    generation_config = GenerationConfig(
        temperature=args.temperature,
        do_sample=True,
        num_return_sequences=1,
        reprtition_penalty=1.05
    )
    
    def prediction(model, infer_dataloader):
        predicted_sequences = []
        sources_sequences = []
        model.eval()

        for step, batch in enumerate(infer_dataloader):
            # TODO, add prompts, choosen, rejected
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            sources_sequences += batch['sources']
            del batch['sources']
            batch = to_device(batch, device)
            progress_bar.update(1)
            prompt_len = batch['input_ids'].shape[1]

            # update progress bar
            if args.global_rank == 0:
                progress_bar.update(1)
                description = f"Step {step}"
                progress_bar.set_description(description, refresh=False)

            with torch.no_grad():
                generate_ids = model.generate(batch['input_ids'],
                                            attention_mask=batch['attention_mask'],
                                            max_new_tokens=args.max_ans_len,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.eos_token_id,
                                            generation_config=generation_config,
                                            use_cache=True
                                            )

            sequences = tokenizer.batch_decode(generate_ids[:, prompt_len:], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            predicted_sequences += sequences

        return sources_sequences, predicted_sequences


    def save_inference_results(sources_sequences: list, predicted_sequences: list, evaluation_aspect, task):
        # save as a json file
        df = {'prompts': sources_sequences, 'results': predicted_sequences}
        output_dir = os.path.join(args.inference_output_path, evaluation_aspect)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(output_dir,task+'.json')
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False)

    # task_dict = {"helpful":['Koala', 'Alpaca', 'OpenAssistant', 'self-instruct', 'LIMA'],
    #              "harmless":['CoNa', 'Malicious', 'Controversial', 'PhysicalSafetyUnsafe']}

    # for faster testing
    task_dict = {"helpful":['self-instruct', 'LIMA'],
                 "harmless":['CoNa']}

    for evaluation_aspect in args.inference_tasks:
    # Prepare the data
        tasks = task_dict[evaluation_aspect]
        evaluation_dir = os.path.join(args.data_path, evaluation_aspect)
        for task in tasks:
            task_dir = os.path.join(evaluation_dir, task)
            data_file = os.path.join(task_dir,"test.json")
            infer_dataset = load_dataset(os.path.join(args.data_path,"data_process.py"),data_file=data_file)
            infer_dataset = infer_dataset['test']

            inf_data_collator = DataCollator(
                tokenizer,
                model=model,
                padding="longest",
                max_prompt_len=args.max_prompt_len,
                pad_to_multiple_of=8,
            )

            infer_sampler = SequentialSampler(infer_dataset)
            infer_dataloader = DataLoader(infer_dataset,
                                        collate_fn=inf_data_collator,
                                        sampler=infer_sampler,
                                        batch_size=args.inference_batch)

            progress_bar = tqdm(total=len(infer_dataloader), leave=True, disable=(args.global_rank != 0))


            # Inference !
            print_rank_0("***** Start inference *****", args.global_rank)
            sources_sequences, predicted_sequences = prediction(model, infer_dataloader)
            save_inference_results(sources_sequences, predicted_sequences, evaluation_aspect, task)



if __name__ == "__main__":
    main()
