import os
import tqdm
import torch
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device


class EPI(CL_Base_Model):

    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args, is_ds_engine=True):
        super(EPI, self).__init__(model, tokenizer, optimizer,
                                  train_task_list, eval_task_list, test_task_list, args)
        # print all attributes of the model (deepspeed engine actually)
        # print_rank_0(f"Deepspeed attributes: {model.__dict__}", self.args.local_rank)

        if is_ds_engine:
            hidden_size = model.module.config.hidden_size
        else:
            hidden_size = model.config.hidden_size

        self.means = []
        self.covs = []
        self.cov_inv = torch.zeros(hidden_size, hidden_size)
        if train_task_list is not None:
            self.task_count = len(train_task_list)

        self.device = torch.device("cuda") if self.args.local_rank == -1 else torch.device("cuda", self.args.local_rank)

    def train_continual(self):
        for task_num, task_name in enumerate(self.train_task_list):
            self.train_one_task(task_name, task_num, int(self.args.num_train_epochs[task_num]))
            # may only do statistic when local_rank == 0
            self.statistic(task_name, task_num)
            self.save_model(task_num)
            # self.expand_lora_adapters(task_num+1)

    def expand_lora_adapters(self, task_num):
        new_task = f"task_{task_num}"
        self.model.peft_config[new_task] = self.model.peft_config["task_0"]
        # from dataclasses import replace
        # self.model.peft_config[new_task] = replace(self.model.peft_config["task_0"])
        self.model.inject_adapter(self.model, new_task)

    def train_one_task(self, task_name, task_num, epochs):
        print_rank_0(f"model architecture: {self.model}", self.args.local_rank)

        # training progress
        train_dataloader = self.train_task_list[task_name]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm.tqdm(total=total_steps, leave=True,
                            disable=(self.args.local_rank != 0))

        # we only active the LoRA params for the current task
        # ! there is a bug that current PEFT is able to
        # ! training the first adapter only
        # ! so we need to set the adapter to the first adapter
        # ! for the any task in the training process
        # ! and copy the adapter weights from the first adapter
        # ! to the current adapter
        # self.model.set_adapter(f"task_{task_num}")
        self.model.set_adapter(f"task_neo")
        self.model.enable_adapter_layers()
        print_rank_0(
            f"Training on task {task_num} only use LoRA adapter {self.model.active_adapters}", self.args.local_rank)
        print_rank_0(f"{trainable_params(self.model)}", self.args.local_rank)

        self.model.train()
        for epoch in range(epochs):
            print_rank_0(
                f"\nBeginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.local_rank)

            # train on one batch
            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                # loss.backward()
                # self.optimizer.step()
                # self.optimizer.zero_grad()

                # for deepspeed
                self.model.backward(loss)
                self.model.step()

                if self.args.local_rank == 0:
                    progress_bar.update(1)
                    description = f"Task {task_num} Epoch {epoch+1}/{epochs} Loss {loss.item(): .4f}"
                    progress_bar.set_description(description, refresh=False)

        print_rank_0(f"Training on task {task_num} is finished", self.args.local_rank)

        print_rank_0(f"Copy the adapter weights from task_neo to task_{task_num}", self.args.local_rank)
        # copy the adapter weights to the current adapter
        state_dict = self.model.state_dict()
        for key in state_dict:
            if f"task_{task_num}" in key:
                neo_key = key.replace(f"task_{task_num}", "task_neo")
                state_dict[key] = state_dict[neo_key]
        self.model.load_state_dict(state_dict)

        print_rank_0(f"Copy the adapter weights from task_neo to task_{task_num} is finished", self.args.local_rank)
        print_rank_0(f"Training on task {task_num} is finished", self.args.local_rank)


    def evaluate(self):
        for task_num, task_name in enumerate(self.train_task_list):
            self.evaluate_one_task(task_name, task_num, self.args.epochs)

    def evaluate_one_task(self, task, task_num):
        pass

    def generate(self, input_ids, attention_mask, **kwargs):
        # we re-implement the generate function
        # for adapt to inference and evaluation scripts

        self.model.eval()
        # step 1: we first get all pre-logits from the model

        # disable all the adapters
        # self.model.set_adapter([])
        self.model.disable_adapter_layers()
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             output_hidden_states=True,
                             use_cache=False)
        # [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]
        # [batch_size, hidden_size]
        prelogits = mean_pooling(last_hidden_state, attention_mask)

        # step 2: we find the preferred adapter for all samples

        score_over_tasks = []
        for task_num, task_mean in enumerate(self.means):
            score = mahalanobis(prelogits, task_mean, self.cov_inv)
            score_over_tasks.append(score)

        # [batch_size, num_tasks]
        score_over_tasks = torch.stack(score_over_tasks, dim=1)
        _, preferred_adapters = score_over_tasks.min(dim=1)

        print_rank_0(f"Score over tasks: {score_over_tasks}", self.args.local_rank)
        print_rank_0(f"Preferred adapters: {preferred_adapters}", self.args.local_rank)

        # step 3: we use the preferred adapter to generate the output
        adopt_answer = [None] * len(preferred_adapters)
        selected_adapters = set(preferred_adapters.tolist())
        self.model.set_adapter(f"task_neo")
        self.model.enable_adapter_layers()

        print_rank_0(f"Current active adapters: {self.model.active_adapters}", self.args.local_rank)

        for adapter in selected_adapters:
            # ! there is a bug here
            # ! LoraModel can only activate the first adapter
            # ! so we copy the params from the desired adapter
            # ! to the first adapter and activate it
            print_rank_0(f"We are using adapter {adapter}", self.args.local_rank)
            print_rank_0(f"Copy the adapter weights from task_neo to task_{adapter}", self.args.local_rank)

            # copy the adapter weights to the current adapter
            state_dict = self.model.state_dict()
            for key in state_dict:
                if "task_neo" in key:
                    task_key = key.replace("task_neo", f"task_{adapter}")
                    state_dict[key] = state_dict[task_key]
            self.model.load_state_dict(state_dict)

            print_rank_0(f"Copy the adapter weights from task_neo to task_{adapter} is finished", self.args.local_rank)

            # self.model.disable_adapter_layers() # disable all the adapters for debug
            outputs = self.model.generate(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          **kwargs)

            for idx, preferred_adapter in enumerate(preferred_adapters):
                if preferred_adapter == adapter:
                    assert adopt_answer[idx] is None
                    adopt_answer[idx] = outputs[idx]

        # match the shape of the adopt_answer
        max_len = max([len(ans) for ans in adopt_answer])
        for idx, ans in enumerate(adopt_answer):
            if len(ans) < max_len:
                pad_token_id = kwargs.get("pad_token_id")
                padding_ans = torch.tensor([pad_token_id] * (max_len - len(ans)), dtype=torch.long, device=ans.device)
                adopt_answer[idx] = torch.cat([ans, padding_ans])

        adopt_answer = torch.stack(adopt_answer, dim=0)
        print_rank_0(f"Adopted adapters: {selected_adapters}", self.args.local_rank)
        print_rank_0(f"Adopted answers: {adopt_answer}", self.args.local_rank)

        return adopt_answer

    def statistic(self, task, task_num):
        if self.args.local_rank > 0:
            return

        # statistic on a single task for mean and covariance

        # disable all the adapters
        self.model.disable_adapter_layers()
        print_rank_0(f"Statistic on task {task_num} (not using any adapter)", self.args.local_rank)
        # print_rank_0(f"Current active adapters: {self.model.active_adapters}", self.args.local_rank)

        self.model.eval()

        train_dataloader = self.train_task_list[task]
        total_steps = len(train_dataloader)
        progress_bar = tqdm.tqdm(total=total_steps, leave=True,
                            disable=(self.args.local_rank != 0))
        with torch.no_grad():
            prelogits = []
            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, self.device)
                # get pre-logits from the base model
                # in this way, we don't need to disable any adapter.
                # this way can be adpot to deepspeed
                outputs = self.model.module.model(**batch,
                                     output_hidden_states=True,
                                     output_attentions=False,
                                     use_cache=False)
                # [batch_size, seq_len, hidden_size]
                last_hidden_state = outputs.hidden_states[-1]
                assert "attention_mask" in batch
                attention_mask = batch["attention_mask"]
                pooling = mean_pooling(last_hidden_state, attention_mask)
                prelogits.extend(pooling.tolist())

                progress_bar.update(1)
                description = f"Task {task_num} Step {step+1}/{total_steps}"
                progress_bar.set_description(description, refresh=False)

        # [num_samples, hidden_size]
        prelogits = torch.tensor(prelogits)

        # [hidden_size]
        task_mean = prelogits.mean(dim=0)
        # [hidden_size, hidden_size]
        task_cov = torch.cov((prelogits - task_mean).T)

        self.means.append(task_mean)
        self.covs.append(task_cov)

        # update the inverse of the covariance matrix
        task_cov_mean = torch.stack(list(self.covs), dim=0).mean(dim=0)
        # self.cov_inv = torch.linalg.inv(task_cov_mean)
        self.cov_inv = torch.linalg.pinv(task_cov_mean, hermitian=True)

    def save_model(self, task_num):
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.local_rank)

        if self.args.local_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(task_num))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)

            # self.model.module.save_pretrained(peft_model_id) # will save the whole model instead of the adapter only
            torch.save(self.model.module.state_dict(), os.path.join(peft_model_id, "pytorch_model.bin"))
            # save the task_count, mean, covariance and inverse covariance matrix also
            mean_cov_path = os.path.join(peft_model_id, "mean_cov.pt")
            torch.save({"task_count": task_num,
                        "means": self.means,
                        "covs": self.covs,
                        "cov_inv": self.cov_inv}, mean_cov_path)
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(
                f'Sucessfully saving the final model to {peft_model_id}', self.args.local_rank)

    def eval(self):
        self.model.eval()

    @staticmethod
    def load_model(model, model_path, args):
        # model: LoraModel
        # model_path: string

        print_rank_0(f"Loading the model from {model_path}")

        # we only need the model in the EPI
        epi = EPI(model, None, None, None, None, None, args, is_ds_engine=False)

        # step 1: load task_count, mean, cov and cov_inv
        mean_cov_path = os.path.join(model_path, "mean_cov.pt")
        mean_cov = torch.load(mean_cov_path)

        device = epi.device
        epi.means = [mean.to(device) for mean in mean_cov["means"]]
        epi.covs = [cov.to(device) for cov in mean_cov["covs"]]
        epi.cov_inv = mean_cov["cov_inv"].to(device)
        epi.task_count = mean_cov["task_count"]

        # step 2: restore LoRA adapters
        
        # model: LoRA model only have one adapter
        # step 2.1: expand the adapter for match model shape to checkpoint
        # ! now the logic is move to the model definition
        # for task_num in range(1, epi.task_count):
        #     new_task = f"task_{task_num}"
        #     epi.model.peft_config[new_task] = epi.model.peft_config["task_0"]
        #     epi.model.inject_adapter(epi.model, f"task_{task_num}")

        # step 2.2: copy the adapter weights
        # model = model.from_pretrained(model_path)
        # model: LoRA model have all the adapters now

        print_rank_0(model, args.local_rank)
        print_rank_0(f"Successfully load the model from {model_path}")
        return epi


def mean_pooling(hidden_states, attention_mask):
    pooled_output = torch.sum(hidden_states * attention_mask.unsqueeze(-1),
                              dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
    return pooled_output


def mahalanobis(querys, mean, cov_inv, norm=2):
    """
    args:
        querys: [n, dim]
        mean: [dim]
        cov_inv: [dim, dim]
    return：
        [n]
    """
    diff = querys - mean
    # [n, dim] = ([n, dim] @ [dim, dim]) * [n, dim] = [n, dim] * [n, dim]
    maha_dis = torch.matmul(diff, cov_inv) * diff

    if norm == 2:
        return maha_dis.sum(dim=1)
    if norm == 1:
        return maha_dis.abs().sqrt().sum(dim=1)
    if norm == 'inf':
        return maha_dis.max(dim=1)

def trainable_params(model):
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total params: {all_params}, Trainable params: {trainable_params}"