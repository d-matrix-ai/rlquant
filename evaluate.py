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
        case "awq":
            raise ValueError("Run awq_compress.py!")
        case "gptq-2":
            return GPTQConfig(
                bits=2,
                dataset=gptq_data,
                tokenizer=tokenizer,
            )
        case "gptq-4":
            return GPTQConfig(
                bits=4,
                dataset=gptq_data,
                tokenizer=tokenizer,
            )
        case "gptq-8":
            return GPTQConfig(
                bits=8,
                dataset=gptq_data,
                tokenizer=tokenizer,
            )
        case "None":
            return None
        case _ :
            raise Exception("incompatible post-training quantization type")

def main():
    args = parse_arguments()

    model_name = (args.base_model_name if args.quantize else args.model_name)
    adpath = args.adapter_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if args.evaluate:
        args.train_dataset=""
    _, eval_data = load_data_for_train_and_eval(args)
    os.environ["WANDB_PROJECT"]=args.wb_project


    if args.ptq:
        reward_fn = Reward(tokenizer, False, True)
        """For now only awq and bnb 4bit/8bit PTQ algorithms supported"""
        if "awq" not in args.ptq_type:
            accelerator=Accelerator()
            quant_config = get_ptq_config(args.ptq_type)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                trust_remote_code=True,
            )
            num_generations = 1
            batch_size = 32
            model.eval()
            AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 8
            if accelerator.is_main_process:
                wandb.init(project=args.wb_project, name= args.wb_run_name)
            tasks_list = ["aime", "amc", "math", "minerva", "olympiad_bench"]
            data = []
            print("Beginning eval cycle")
            collate_fn = Collate_Fn(tokenizer)
            for task in tasks_list:
                dataloader = DataLoader(eval_data[task], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                dataloader = accelerator.prepare(dataloader)
                scores = []
                for prompts, inputs, answers in tqdm(dataloader):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    all_generations_per_prompt = [[] for _ in range(len(prompts))]

                    for _ in range(num_generations):
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=1.0,
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
                    print(f"eval/reward_{task}: {mean_reward},\n eval/std_{task}: {mean_stdev}")
                    wandb.log({"task": task, f"eval/reward_{task}": mean_reward, f"eval/std_{task}": mean_stdev})
            if accelerator.is_main_process:
                print(f"eval/reward: {np.mean(np.array(data))},\n eval/std: {np.std(np.array(data))}")
                wandb.log({"eval/reward": np.mean(np.array(data)), "eval/reward_std": np.std(np.array(data))})
            print("Evaluation done!")
        else:
            model = LLM(model_name, tensor_parallel_size=2)
            # if dist.get_rank()==0:
            wandb.init(project=args.wb_project, name= args.wb_run_name)
            tasks_list = ["aime", "amc", "math", "minerva", "olympiad_bench"]
            data=[]
            for task in tasks_list:
                e_data = eval_data[task]
                inputs = []
                answers = []
                for item in e_data:
                    inputs.append(item["problem"])
                    answers.append(item["answer"])
                sampling_params = SamplingParams(temperature=1.0,top_p=1.0, max_tokens=512)
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
                print(f"eval/reward_{task}: {mean_reward},\n eval/std_{task}: {mean_stdev}")
                wandb.log({"task": task, f"eval/reward_{task}": mean_reward, f"eval/std_{task}": mean_stdev})            
            print(f"eval/reward: {np.mean(np.array(data))},\n eval/std: {np.std(np.array(data))}")
            wandb.log({"eval/reward": np.mean(np.array(data)), "eval/reward_std": np.std(np.std(data))})
            print("Evaluation done!")
        return
    elif args.quantize:                                      
        quant_config = get_quant_config(args.ptq_type)         
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda", 
            quantization_config=quant_config,
            trust_remote_code=True,
        )
        base_model.config.use_cache = False
        
        p_model = PeftModel.from_pretrained(base_model,args.adapter_path)
        model = p_model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cuda", 
                trust_remote_code=True,
            )
    reward_fn = Reward(tokenizer, False, False)
    training_args = GRPOConfig(
        output_dir=f"./{args.new_model_name}",
        report_to="wandb",
        run_name=args.wb_run_name,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        bf16=True,
        use_vllm=False,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        generation_batch_size=32, #Added                 --- try 64 as well.
        gradient_accumulation_steps=2, # Added
        max_prompt_length=int(args.max_prompt_length),
        max_completion_length=int(args.max_completion_length),
        gradient_checkpointing=args.gc,
        logging_steps=1,
        eval_steps=1,
        num_train_epochs=1,
        num_generations=8, 
        save_total_limit=5,
        save_strategy='steps',
        eval_strategy='steps',
        max_grad_norm=1.0,
        beta=0,
        remove_unused_columns=False, 
        save_steps=20,
        top_p=1,
        temperature=1, 
        logging_first_step=True,
        loss_type="grpo",
        eval_on_start=True,
        do_train=False,
    )
    if args.adam8:
        training_args.optim="adamw_8bit"

    if args.ptq and not args.quantize:
        evaluate_model(model, tokenizer, eval_dataset, reward_fn, training_args)
    else:
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            eval_dataset= eval_data,
            processing_class=tokenizer,
            reward_funcs=[reward_fn], 
        )

        trainer.evaluate()



if __name__=="__main__":
    main()
