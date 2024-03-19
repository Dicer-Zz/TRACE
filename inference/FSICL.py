import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import torch
import random
from tqdm import tqdm
import json
import argparse
from tqdm import tqdm
import json
import deepspeed
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler

import torch.nn.functional as F
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, set_random_seed, load_hf_tokenizer
from utils.model.model_utils import create_hf_model
from training.params import AllDatasetName
from evaluations import (
    eval_ScienceQA,
    eval_MeetingBank,
    eval_PapyrusF,
    eval_CStance,
    eval_Py150,
    eval_FOMC,
    eval_NumGLUE_cm,
    eval_NumGLUE_ds,
    eval_20Minuten,
)  # to be continued


# os.environ['CUDA_VISIBLE_DEVICES']="0"

EXAMPLE_PROMPT = {
    "EN": "We will give you several examples and you should follow them to accomplish the task.\n Examples:\n",
    "ZH": "我们将给出一些例子，您需要按照这些例子来完成任务。\n 例子：\n",
    "DE": "Wir geben Ihnen einige Beispiele und Sie sollten ihnen folgen, um die Aufgabe zu erledigen.\n Beispiele:\n",
}

COT_PROMPT = {
    "EN": "\nLet's think step by step:\n",
    "ZH": "\n让我们一步一步地思考：\n",
    "DE": "\nLassen Sie uns Schritt für Schritt denken:\n",
}

COT_ANSWER = {
    "EN": "\nGiven the above question and reasoning, the answer is:\n",
    "ZH": "\n根据上述问题和推理，答案是：\n",
    "DE": "\nAngesichts der obigen Frage und Überlegung ist die Antwort:\n",
}

TASK_LANG = {
    "FOMC": "EN",
    "C-STANCE": "ZH",
    "ScienceQA": "EN",
    "NumGLUE-cm": "EN",
    "NumGLUE-ds": "EN",
    "MeetingBank": "EN",
    "Py150": "EN",
    "20Minuten": "DE",
}

TASK_INSTRUCTIONS = {
    "FOMC": "What is the monetary policy stance for the following text? A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\n",
    "C-STANCE": "判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n",
    "ScienceQA": "Choose an answer for the following question and give your reasons.\n",
    "NumGLUE-cm": "Solve the following math problem.\n",
    "NumGLUE-ds": "Solve the following math problem.\n",
    "MeetingBank": "Write a summary of the following meeting transcripts.\n",
    "Py150": "Continue writing the code.\n",
    "20Minuten": "Provide a simplified version of the following paragraph in German.\n",
}

SELF_CONSISTENCY_TASK = ["ScienceQA", "FOMC", "C-STANCE", "NumGLUE-cm", "NumGLUE-ds"]


def parse_args():
    def list_of_strings(arg):
        return arg.split(",")

    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Dahoas/rm-static",
        help="Path to the training dataset. A single data path.",
    )
    parser.add_argument(
        "--dataset_name",
        type=list_of_strings,
        default="all",
        help="Dataset to be used.",
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files/",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        default=0.7,
        help="Generate temperature params.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for decoding.",
    )
    parser.add_argument(
        "--inference_batch",
        type=int,
        default=4,
        help="Inference batch size.",
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )

    # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
    # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )

    # added by wangxiao
    parser.add_argument(
        "--inference_output_path",
        type=str,
        default=None,
        help="Where to store inference results.",
    )

    # In-cotext Learning args
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Number of shots for in-context learning",
    )
    parser.add_argument(
        "--cot",
        action='store_true',
        help="Whether to use chain of thought",
    )
    parser.add_argument(
        "--self_consistency",
        action='store_true',
        help="Whether to use self-consistency",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=0,
        help="Number of paths for self-consistency",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

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

    def append_demos_to_batch(batch, demos, task):
        # batch: {input_ids, attention_mask}
        if not demos:
            # TODO: retrieval-based sampling
            demos = get_random_demonstrations(
                int(args.shots),
                infer_dataset,
                length_limit,
                task,
            )
            # py150 has no instruction
            if task != 'Py150':
                # remove instruction for other tasks
                for i in range(len(demos)):
                    demos[i]["prompt"] = demos[i]["prompt"][len(TASK_INSTRUCTIONS[task]):]
        # print_rank_0("demonstrations length:{}".format(len(demonstrations)), args.global_rank)
        # we sample the demonstrations from the dataset
        # for each batch instead of each sample
        demos_input_ids = tokenizer(TASK_INSTRUCTIONS[task] + EXAMPLE_PROMPT[TASK_LANG[task]] + ''.join(demo["prompt"] + ' ' + demo["answer"] + '\n\n' for demo in demos), return_tensors="pt", padding=True).input_ids
        demos_attention_mask = torch.ones_like(demos_input_ids)
        demos_input_ids = demos_input_ids.to(device)
        demos_attention_mask = demos_attention_mask.to(device)
        # expand the demonstrations to the batch size
        demos_input_ids = demos_input_ids.repeat(batch["input_ids"].shape[0], 1)
        demos_attention_mask = demos_attention_mask.repeat(batch["input_ids"].shape[0], 1)
        # append demonstrations to batch
        batch["input_ids"] = torch.cat([demos_input_ids, batch["input_ids"]], dim=1)
        batch["attention_mask"] = torch.cat([demos_attention_mask, batch["attention_mask"]], dim=1)
        return batch

    def prediction(model, infer_dataloader, task):
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        model.eval()

        for step, batch in enumerate(infer_dataloader):
            # sources_sequences += batch['sources']
            ground_truths += batch["gts"]
            del batch["sources"]
            del batch["gts"]
            batch = to_device(batch, device)

            # append demonstrations to batch
            if args.shots > 0:
                batch = append_demos_to_batch(batch, None, task)

            # update progress bar
            if args.global_rank == 0:
                progress_bar.update(1)
                description = f"Step {step}"
                progress_bar.set_description(description, refresh=False)

            with torch.no_grad():
                # sft config
                if args.self_consistency and task in SELF_CONSISTENCY_TASK:
                    generation_config = GenerationConfig(
                        temperature=args.temperature,
                        do_sample=True,
                        num_return_sequences=args.paths,
                        repetition_penalty=args.repetition_penalty,
                    )
                else:
                    generation_config = GenerationConfig(
                        temperature=args.temperature,
                        do_sample=True,
                        num_return_sequences=1,
                        repetition_penalty=args.repetition_penalty,
                    )

                if args.cot and task in SELF_CONSISTENCY_TASK:
                    cot_prompt_input_ids = tokenizer(COT_PROMPT[TASK_LANG[task]], return_tensors="pt").input_ids
                    cot_prompt_input_ids = cot_prompt_input_ids.to(device)
                    # expand the COT_PROMPT to the batch size
                    cot_prompt_input_ids = cot_prompt_input_ids.repeat(batch["input_ids"].shape[0], 1)
                    batch["input_ids"] = torch.cat([batch["input_ids"], cot_prompt_input_ids], dim=1)
                    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(cot_prompt_input_ids)], dim=1)

                max_seq_len = batch["input_ids"].shape[1]
                print_rank_0(f"input_ids shape: {batch['input_ids'].shape}", args.global_rank)
                print_rank_0(f"attention_mask shape: {batch['attention_mask'].shape}", args.global_rank)

                # print_rank_0(f"input_ids: {batch['input_ids']}", args.global_rank)
                # print_rank_0(f"attention_mask: {batch['attention_mask']}", args.global_rank)

                generate_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=args.max_ans_len,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.unk_token_id,
                    generation_config=generation_config,
                    use_cache=True,
                )

                # cot
                if args.cot and task in SELF_CONSISTENCY_TASK:
                    # append the COT_ANSWER to the end of the prompt
                    # generate_ids: [batch_size * paths, length]
                    cot_ids = tokenizer(COT_ANSWER[TASK_LANG[task]], return_tensors="pt").input_ids
                    cot_ids = cot_ids.to(device)
                    cot_ids = cot_ids.repeat(generate_ids.shape[0], 1)
                    generate_ids = torch.cat([generate_ids, cot_ids], dim=1)
                    max_seq_len = generate_ids.shape[1]

                    print_rank_0(f"generate_ids shape: {generate_ids.shape}", args.global_rank)

                    # re-generate
                    if args.self_consistency and task in SELF_CONSISTENCY_TASK:
                        generation_config2 = GenerationConfig(
                            temperature=args.temperature,
                            do_sample=True,
                            num_return_sequences=1,
                            repetition_penalty=args.repetition_penalty,
                        )
                        print_rank_0(f"generation config: {generation_config2}", args.global_rank)
                        tmp_ids =[]
                        for i in range(args.paths):
                            tmp_ids.append(model.generate(
                                input_ids=generate_ids[i::args.paths],
                                # attention_mask=batch["attention_mask"],
                                max_new_tokens=args.max_ans_len,
                                bos_token_id=tokenizer.bos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.unk_token_id,
                                generation_config=generation_config2,
                                use_cache=True,
                            ))
                        print_rank_0(f"tmp_ids shape: {[ids.shape for ids in tmp_ids]}", args.global_rank)
                        # match the length of the generated sequences
                        max_length = max([ids.shape[1] for ids in tmp_ids])
                        for i in range(args.paths):
                            if tmp_ids[i].shape[1] < max_length:
                                pad_len = max_length - tmp_ids[i].shape[1]
                                pad_ids = torch.full((tmp_ids[i].shape[0], pad_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
                                tmp_ids[i] = torch.cat([tmp_ids[i], pad_ids], dim=1)

                        generate_ids = torch.cat(tmp_ids, dim=0)
                    else:
                        generate_ids = model.generate(
                            input_ids=generate_ids,
                            # attention_mask=batch["attention_mask"],
                            max_new_tokens=args.max_ans_len,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.unk_token_id,
                            generation_config=generation_config,
                            use_cache=True,
                        )

                    print_rank_0(f"generate_ids shape: {generate_ids.shape}", args.global_rank)
                    # print_rank_0(f"generate_ids: {generate_ids}", args.global_rank)

            if args.global_rank <= 0:
                # same special decoding trick
                sou_sequences = tokenizer.batch_decode(
                    generate_ids[:, :max_seq_len],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                # if task == "FOMC" or task == "C-STANCE":
                #     pre_sequences = tokenizer.batch_decode(
                #         generate_ids[:, max_seq_len : max_seq_len + 1],
                #         skip_special_tokens=True,
                #         clean_up_tokenization_spaces=False,
                #     )
                # else:

                pre_sequences = tokenizer.batch_decode(
                    generate_ids[:, max_seq_len:],
                    # generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                # print_rank_0(f"pre_sequnces: {pre_sequences}", args.global_rank)

                # if "NumGLUE" in task:
                #     for i in range(len(pre_sequences)):
                #         pre_sequences[i] = pre_sequences[i].split("\n")[0]
                # elif "MeetingBank" in task:
                #     for i in range(len(pre_sequences)):
                #         pre_sequences[i] = pre_sequences[i].split(
                #             "Meeting transcripts"
                #         )[0]
                # elif "ScienceQA" in task:
                #     for i in range(len(pre_sequences)):
                #         pre_sequences[i] = pre_sequences[i].split("Question:")[0]
                # elif "Py150" in task:
                #     for i in range(len(pre_sequences)):
                #         pre_sequences[i] = pre_sequences[i].split("<EOL>")[0]
                # elif "20Minuten" in task:
                #     for i in range(len(pre_sequences)):
                #         pre_sequences[i] = pre_sequences[i].split("Paragraph")[0]

                if args.self_consistency and task in SELF_CONSISTENCY_TASK:
                    # pre_sequences: [batch_size * paths]
                    # [1, 2, ..., batch_size, 1, 2, ..., batch_size, ...]
                    # list of string
                    # vote for the final result
                    
                    # we only need the answer
                    pre_sequences = [seq.strip()[:1] for seq in pre_sequences]

                    batch_size = len(pre_sequences) // args.paths
                    pre_sequences = [pre_sequences[i*batch_size:(i+1)*batch_size] for i in range(args.paths)]
                    pre_sequences = list(map(list, zip(*pre_sequences)))
                    pre_sequences = [max(set(seq), key=seq.count) for seq in pre_sequences]

                predicted_sequences += pre_sequences
                sources_sequences += sou_sequences

                # # early stop
                # if step == 5:
                #     break

        return sources_sequences, predicted_sequences, ground_truths

    def get_random_demonstrations(dem_num, infer_dataset, length_limit, task):
        demonstrations = []
        total_length = 0
        i = 0
        tmp = 0
        while i < dem_num:
            tmp += 1
            if tmp == 10000:
                break
            demonstration_id = random.randint(0, len(infer_dataset) - 1)
            demonstration = infer_dataset[demonstration_id]

            if (
                total_length
                + len(tokenizer(demonstration["prompt"])["input_ids"])
                + len(tokenizer(demonstration["answer"])["input_ids"])
                <= length_limit
            ):
                demonstrations.append(demonstration)
                i += 1

        return demonstrations

    # set evaluation batch size
    # only support bs = 1, cause right padding training logic

    if args.dataset_name[0] == "all":
        Datasets = AllDatasetName
    else:
        Datasets = args.dataset_name

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    assert tokenizer.padding_side == "left"
    assert tokenizer.truncation_side == "left"

    model_class = AutoModelForCausalLM

    model = create_hf_model(
        model_class,
        args.model_name_or_path,
        tokenizer,
        ds_config=None,
        torch_dtype=torch.bfloat16,
    )

    # replace_with_kernel_inject = False if "falcon" in args.model_name_or_path.lower() else True
    replace_with_kernel_inject = False
    ds_engine = deepspeed.init_inference(
        model,
        mp_size=world_size,
        dtype=torch.bfloat16,
        checkpoint=None,
        replace_with_kernel_inject=replace_with_kernel_inject,
        max_out_tokens=args.max_prompt_len + args.max_ans_len,
    )
    model = ds_engine.module

    for task in Datasets:
        data_path = args.data_path
        inference_output_path = args.inference_output_path
        inference_output_path = os.path.join(inference_output_path, task)

        dataset_path = os.path.join(data_path, task)

        _, _, infer_dataset = create_prompt_dataset(
            args.local_rank, dataset_path, args.data_output_path, args.seed
        )

        length_limit = args.max_prompt_len - len(
            tokenizer(TASK_INSTRUCTIONS[task] + EXAMPLE_PROMPT[TASK_LANG[task]])["input_ids"]
        )

        if args.cot and task in SELF_CONSISTENCY_TASK:
            length_limit -= len(tokenizer(COT_PROMPT[TASK_LANG[task]] + COT_ANSWER[TASK_LANG[task]])["input_ids"])

        # demonstrations = get_random_demonstrations(
        #     int(args.shots),
        #     infer_dataset,
        #     length_limit,
        #     task,
        # )
        # print_rank_0(
        #     "demonstrations length:{}".format(len(demonstrations)), args.global_rank
        # )
        # # avoid using demonstrations for MeetingBank
        # if task == "MeetingBank":
        #     demonstrations = []

        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=True,
            # demonstrations=demonstrations,
            task=task,
        )

        infer_sampler = SequentialSampler(infer_dataset)
        infer_dataloader = DataLoader(
            infer_dataset,
            collate_fn=inf_data_collator,
            sampler=infer_sampler,
            batch_size=args.inference_batch,
        )

        progress_bar = tqdm(total=len(infer_dataloader), leave=True)
        print_rank_0("***** Start inference *****", args.global_rank)
        sources_sequences, predicted_sequences, ground_truths = prediction(
            model, infer_dataloader, task
        )

        if task == "ScienceQA":
            evaluation_result = eval_ScienceQA.eval(predicted_sequences, ground_truths)
        elif task == "MeetingBank":
            evaluation_result = eval_MeetingBank.eval(
                predicted_sequences, ground_truths
            )
        elif task == "C-STANCE":
            evaluation_result = eval_CStance.eval(predicted_sequences, ground_truths)
        elif task == "Papyrus-f":
            evaluation_result = eval_PapyrusF.eval(predicted_sequences, ground_truths)
        elif task == "Py150":
            evaluation_result = eval_Py150.eval(predicted_sequences, ground_truths)
        elif task == "FOMC":
            evaluation_result = eval_FOMC.eval(predicted_sequences, ground_truths)
        elif task == "NumGLUE-cm":
            evaluation_result = eval_NumGLUE_cm.eval(predicted_sequences, ground_truths)
        elif task == "NumGLUE-ds":
            evaluation_result = eval_NumGLUE_ds.eval(predicted_sequences, ground_truths)
        elif task == "20Minuten":
            evaluation_result = eval_20Minuten.eval(
                sources_sequences, predicted_sequences, ground_truths
            )

        print(evaluation_result)
        df = {
            "eval": evaluation_result,
            "prompts": sources_sequences,
            "results": predicted_sequences,
            "labels": ground_truths,
        }

        if not os.path.exists(inference_output_path):
            os.makedirs(inference_output_path)

        with open(
            inference_output_path + "/results-" + task + ".json", "w+", encoding="utf-8"
        ) as file:
            json.dump(df, file, ensure_ascii=False)


if __name__ == "__main__":
    main()
