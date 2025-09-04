import os
import argparse 

from datasets import load_from_disk
from datasets import DatasetDict



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from transformers import BitsAndBytesConfig, GPTQConfig, Trainer
from trl import GRPOTrainer, GRPOConfig, TrlParser, ModelConfig, get_peft_config
from transformers import BitsAndBytesConfig, get_polynomial_decay_schedule_with_warmup
from transformers.utils import logging
logger = logging.get_logger("transformers")
import wandb

from vllm import LLM, SamplingParams

from helper_functions import boxed_reward_fn
from train_model import Reward, get_quant_config, load_data_for_train_and_eval, parse_arguments

from peft import LoraConfig, get_peft_model, PeftModel

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import numpy as np
import statistics

from accelerate import Accelerator
from accelerate.state import AcceleratorState
import torch.distributed as dist

from awq_compress import compress_model

class Collate_Fn():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @property
    def __name__(self) -> str:
        return self.__class__.__name__
    
    def __call__(self, batch):
        prompts = [x["problem"] for x in batch]
        answers = [x["answer"] for x in batch]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        return prompts, inputs, answers


def get_ptq_config(ptq_type: str, tokenizer=None, gptq_data=None):
    match ptq_type:
        case "bnb-8bit":
            return BitsAndBytesConfig(load_in_8bit=True,)
        case "bnb-4bit":
            return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
        case _ :
            raise Exception("incompatible post-training quantization type")


def accelerate_evaluate(model_name, reward_fn, args, tokenizer, eval_data):
    accelerator=Accelerator()
    if args.quantize and args.qtype != "8bit":                             
        quant_config = get_quant_config(args.qtype)         
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            trust_remote_code=True,
        )
        #base_model.config.use_cache = False
        #model = PeftModel.from_pretrained(base_model,args.adapter_path).to(device="cuda")
        # model = p_model.merge_and_unload()

    elif args.ptq:
        quant_config = get_ptq_config(args.ptq_type)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    model.eval()
    
    AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 8
    
    if accelerator.is_main_process:
        wandb.init(project=args.wb_project, name= args.wb_run_name)
    
    tasks_list = ["aime", "amc", "math", "minerva", "olympiad_bench"]
    data = []
    print("Beginning eval cycle")
    collate_fn = Collate_Fn(tokenizer)
    for task in tasks_list:
        dataloader = DataLoader(eval_data[task], batch_size=32, shuffle=True, collate_fn=collate_fn)
        dataloader = accelerator.prepare(dataloader)
        scores = []
        for prompts, inputs, answers in tqdm(dataloader):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            all_generations_per_prompt = [[] for _ in range(len(prompts))]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_completion_length,
                    do_sample=False,
                    #temperature=1.0,
                    top_p=1.0,
                )
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i in range(len(prompts)):
                all_generations_per_prompt[i].append(generated_texts[i])
        
        for prompt, gen, ref in zip(prompts, all_generations_per_prompt, answers):
            gen_scores = reward_fn([prompt], gen, ans=[ref])
            scores.append(gen_scores) 
        all_scores = accelerator.gather_for_metrics(scores)
        data.append(all_scores)
        mean_reward = np.mean(np.array(all_scores))
        mean_stdev = np.std(np.array(all_scores))
        if accelerator.is_main_process:
            print(f"eval_t0/reward_{task}: {mean_reward},\n eval_t0/std_{task}: {mean_stdev}")
            wandb.log({"task": task, f"eval_t0/reward_{task}": mean_reward, f"eval_t0/std_{task}": mean_stdev})
    if accelerator.is_main_process:
        print(f"eval_t0/reward: {np.mean(np.array(data))},\n eval_t0/std: {np.std(np.array(data))}")
        wandb.log({"eval_t0/reward": np.mean(np.array(data)), "eval_t0/reward_std": np.std(np.array(data))})
    print("Evaluation done!")
    


def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = parse_arguments()

    model_name = (args.base_model_name if args.quantize else args.model_name)
    adpath = args.adapter_path
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if args.evaluate:
        args.train_dataset=""
    _, eval_data = load_data_for_train_and_eval(args)
    
    os.environ["WANDB_PROJECT"]=args.wb_project

    """Cannot use trainer with lora models, and evaluation is faster with vllm."""
    reward_fn = Reward(tokenizer, False, True)
    if False:
        """Cannot use vllm with lora models -- unless merging adapter into model, which worsens performance. Can use this with awq, but takes longer."""
        """When using accelerate launch script with `accelerate launch evaluate.py ...` """
        accelerate_evaluate(model_name, reward_fn, args, tokenizer, eval_data)
    else:
        """When using vllm launch script with python evaluate.py ..."""
        print("=====================IN ELSE==============================")
        if args.ptq:
            compressed_model= compress_model(args)
            model = LLM(compressed_model)
        elif args.quantize:
            print("=====================IN LOAD LLM==============================")
            model = LLM(model_name, quantization="bitsandbytes")
        else:
            model = LLM(args.model_name)
        print("=====================MODEL LOADED==============================")
        wandb.init(project=args.wb_project, name= args.wb_run_name)
        tasks_list = ["aime", "amc", "math", "minerva", "olympiad_bench"]
        data=[]
        #model.eval()
        for task in tasks_list:
            e_data = eval_data[task]
            inputs = []
            answers = []
            for item in e_data:
                inputs.append(item["problem"])
                answers.append(item["answer"])
            sampling_params = SamplingParams(temperature=float(args.eval_temp), max_tokens=int(args.max_completion_length))
            outputs = model.generate(inputs, sampling_params)
            scores = []
            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                prompt = inputs[i]
                ref = answers[i]
                score = reward_fn([prompt], [generated_text], ans=[ref])
                scores.append(score)
            data.extend(scores)
            mean_reward = np.mean(np.array(scores))
            mean_stdev = np.std(np.array(scores))
            print(f"eval_t0/reward_{task}: {mean_reward},\n eval_t0/std_{task}: {mean_stdev}")
            wandb.log({"task": task, f"eval_t0/reward_{task}": mean_reward, f"eval_t0/std_{task}": mean_stdev})            
        print(f"eval_t0/reward: {np.mean(np.array(data))},\n eval_t0/std: {np.std(np.array(data))}")
        wandb.log({"eval_t0/reward": np.mean(np.array(data)), "eval_t0/reward_std": np.std(np.std(data))})
        print("Evaluation done!")



if __name__=="__main__":
    main()
