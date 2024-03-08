import os
import tqdm
import torch
from peft import LoraModel
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device


class EPI(CL_Base_Model):

    def __init__(self, model: LoraModel, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args):
        super(EPI, self).__init__(model, tokenizer, optimizer,
                                  train_task_list, eval_task_list, test_task_list, args)
        hidden_size = model.model.config.hidden_size
        self.means = torch.nn.ParameterList()
        self.covs = torch.nn.ParameterList()
        self.cov_inv = torch.nn.Parameter(torch.zeros(
            hidden_size, hidden_size), requires_grad=False)

        self.device = torch.device(
            "cuda") if self.args.local_rank == -1 else torch.device("cuda", self.args.local_rank)

    def train_continual(self):
        for task_num, task_name in enumerate(self.train_task_list):
            self.train_one_task(task_name, task_num, int(self.args.num_train_epochs[task_num]))
            self.statistic(task_name, task_num)
            self.save_model(task_num)
            self.expand_lora_adapters(task_num+1)

    def expand_lora_adapters(self, task_num):
        new_task = f"task_{task_num}"
        self.model.peft_config[new_task] = self.model.peft_config["task_0"]
        self.model.inject_adapter(self.model, f"task_{task_num}")

    def train_one_task(self, task_name, task_num, epochs):
        # training progress
        train_dataloader = self.train_task_list[task_name]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm.tqdm(total=total_steps, leave=True,
                            disable=(self.args.global_rank != 0))

        # we only active the LoRA params for the current task
        self.model.set_adapter(f"task_{task_num}")
        print_rank_0(
            f"Training on task {task_num} only use LoRA adapter {self.model.active_adapters}")

        self.model.train()
        for epoch in range(epochs):
            print_rank_0(
                f"\nBeginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)

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

                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    description = f"Task {task_num} Epoch {epoch+1}/{epochs} Loss {loss.item(): .4f}"
                    progress_bar.set_description(description, refresh=False)

    def evaluate(self):
        for task_num, task_name in enumerate(self.train_task_list):
            self.evaluate_one_task(task_name, task_num, self.args.epochs)

    def evaluate_one_task(self, task, task_num):
        pass

    def generate(self, input_ids, attention_mask, **kwargs):
        # we re-implement the generate function
        # for adapt to inference and evaluation scripts

        # step 1: we first get all pre-logits from the model

        # disable all the adapters
        self.model.set_adapter([])
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             use_cache=False)
        # [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]
        # [batch_size, hidden_size]
        prelogits = mean_pooling(last_hidden_state, attention_mask)

        # step 2: we find the preferred adapter for all samples

        score_over_tasks = []
        for task_num, task_mean in enumerate(self.means):
            score = mahalanobis(prelogits, self.means, self.cov_inv)
            score_over_tasks.append(score)

        # [batch_size, num_tasks]
        score_over_tasks = torch.stack(score_over_tasks, dim=1)
        _, preferred_adapters = score_over_tasks.min(dim=1)

        # step 3: we use the preferred adapter to generate the output
        adopt_answer = [None] * len(preferred_adapters)
        selected_adapters = set(preferred_adapters.tolist())
        for adapter in selected_adapters:
            self.model.set_adapter(f"task_{adapter}")
            outputs = self.model.generate(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          **kwargs)

            for idx, preferred_adapter in enumerate(preferred_adapters):
                if preferred_adapter == adapter:
                    assert adopt_answer[idx] is None
                    adopt_answer[idx] = outputs

        return adopt_answer

    def statistic(self, task, task_num):
        # statistic on a single task for mean and covariance

        # disable all the adapters
        self.model.set_adapter([])
        print_rank_0(f"Statistic on task {task_num} (not using any adapter)")

        self.model.eval()

        with torch.no_grad():
            prelogits = []
            for step, batch in enumerate(self.eval_task_list[task]):
                del batch['sources']
                batch = to_device(batch, self.device)
                outputs = self.model(**batch,
                                     output_hidden_states=True,
                                     output_attentions=False,
                                     use_cache=False)
                # [batch_size, seq_len, hidden_size]
                last_hidden_state = outputs.hidden_states[-1]
                assert "attention_mask" in batch
                attention_mask = batch["attention_mask"]
                pooling = mean_pooling(last_hidden_state, attention_mask)
                prelogits.extend(pooling.tolist())

        prelogits = torch.tensor(prelogits)

        task_mean = prelogits.mean(dim=0)
        task_cov = torch.cov((prelogits - task_mean).T)

        self.means.append(task_mean)
        self.covs.append(task_cov)

        # update the inverse of the covariance matrix
        task_cov_mean = task_cov.mean()
        # self.cov_inv = torch.linalg.inv(task_cov_mean)
        self.cov_inv = torch.linalg.pinv(task_cov_mean, hermitian=True)

    def save_model(self, task_num):
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(task_num))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(
                f'Sucessfully saving the final model to {peft_model_id}', self.args.global_rank)


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
