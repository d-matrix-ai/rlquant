import os
import torch
from datasets import load_from_disk
from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig, TrlParser, ModelConfig, get_peft_config
from helper_functions import boxed_reward_fn

from transformers import BitsAndBytesConfig, get_polynomial_decay_schedule_with_warmup
import argparse 
from transformers.utils import logging
logger = logging.get_logger("transformers")

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, PeftModel
import torch

# /home/ubuntu/.cache/huggingface/accelerate/default_config.yaml
class Reward:
    def __init__(self, tokenizer, use_dense, evaluate):
        self.tokenizer = tokenizer
        self.use_dense = use_dense
        self.ptq_evaluate = evaluate
    
    @property
    def __name__(self) -> str:
        return self.__class__.__name__
    
    def __call__(self, problem, completions, ans=None, **dataset_cols):
        answers = ( ans if self.ptq_evaluate else dataset_cols["answer"] )
        logger.info(f"Prompt: {problem}\n")
        logger.info(f"Completion: {completions}\n")
        logger.info(f"Answer: {answers}\n")
        scores = []
        # Scores returns as an array of ({'formatted':True/False}, 0.0/1.0)
        # We only need an array of the 1.0/0.0 floats for correctness
        for i in range(len(completions)):
            if len(answers)==1 and len(completions) > 1:
                r = boxed_reward_fn(completions[i], answers[0])
                if r[0]['formatted']==True:
                    scores.append(float(r[1] + 0.1))
                else:
                    scores.append(float(r[1]))
            elif i < len(answers):
                if self.use_dense:
                    # Problem with pseudo-dense rewards --- GRPO Trainer expects exactly the same value as tokenizer(completions[i]). Torch.tensor assumes
                    # shortest length for all inputs -- only work around is 1 generation per prompt or overloading GPRO Trainer -- currently not worth it.
                    r = boxed_reward_fn(completions[i], answers[i])
                    ntokens = self.tokenizer(completions[i], return_tensors="pt")["input_ids"][0]
                    dense = [0.0] * len(ntokens)
                    if r[0]['formatted'] ==True:
                        dense[-1] = float(r[1] + 0.1)  #Added offset to incentivize formatting (formatting and correctness are not independent)
                    else:
                        dense[-1] = float(r[1])
                    scores.append(dense)
                else:
                    r = boxed_reward_fn(completions[i], answers[i])
                    if r[0]['formatted']==True:
                        scores.append(float(r[1] + 0.1))
                    else:
                        scores.append(float(r[1]))
        return scores


def load_data_for_train_and_eval(args):
    dataset_name = args.train_dataset 
    eval_name = args.eval_dataset

    dataset = (load_from_disk(dataset_name) if dataset_name != "" else None)
    e_data = load_from_disk(eval_name)

    if dataset is not None:
        prompts_data = dataset["train"].select(range(min(10000, len(dataset["train"]))))
        if args.qwen_math:
            prompts_data = prompts_data.map(lambda example: 
                        {"prompt" : "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}. \
                            <|im_end|>\n<|im_start|>user\n" + example["problem"] +  "<|im_end|>\n<|im_start|>assistant\n"
                        }
            )
        else:
            prompts_data = prompts_data.map(lambda example: {"prompt" : example["problem"]})   
    else:
        prompts_data = None

    tasks_list = ["aime", "amc", "math", "minerva", "olympiad_bench"]
    eval_split = {}
    for task in tasks_list:
        eval_split[task] = e_data[task].shuffle(42)
        if dataset is not None:
            eval_split[task] = eval_split[task].select(range(int(args.eval_num_instances_per_split)))
    eval_data = DatasetDict(eval_split)
    if args.qwen_math:
        eval_data = eval_data.map(lambda example: 
                    {"prompt" : "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}. \
                        <|im_end|>\n<|im_start|>user\n" + example["problem"] +  "<|im_end|>\n<|im_start|>assistant\n"
                    }
        )
    else:
        eval_data = eval_data.map(lambda example: {"prompt": example["problem"]})

    return prompts_data, eval_data


def get_train_args(new_model_name, args):
    lr = 0.000001
    if args.quantize:
        lr=0.0001
    training_args = GRPOConfig(
        output_dir=f"./{new_model_name}",
        # auto_find_batch_size=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=8, # Added
        max_prompt_length=int(args.max_prompt_length),
        max_completion_length=int(args.max_completion_length),
        gradient_checkpointing=args.gc,
        learning_rate=lr,
        logging_steps=int(args.logging_steps),
        eval_steps=int(args.eval_steps),
        num_train_epochs=1,
        num_generations=8, 
        save_total_limit=2,
        save_strategy='steps',
        eval_strategy='steps',
        max_grad_norm=1.0,
        beta=0,
        remove_unused_columns=False, 
        bf16=True,
        report_to="wandb",
        run_name=args.wb_run_name,
        save_steps=int(args.save_steps),
        top_p=1,
        temperature=1, 
        lr_scheduler_type = "linear",
        logging_first_step=True,
        warmup_steps=1,  # to get a training set baseline
        loss_type="grpo",
        eval_on_start=True,
    )
    if args.adam8:
        training_args.optim="adamw_8bit"
    if args.ft_done or not (args.quantize or args.gc or args.torch_oom):
        training_args.use_vllm = True
        training_args.vllm_mode="colocate"
        training_args.vllm_gpu_memory_utilization=0.35
    if args.drgrpo:
        training_args.scale_rewards = False

    return training_args


def get_quant_config(qtype: str):
    match qtype:
        case "4bit":
            return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
        case "None":
            return None


def quant_grad(grad:torch.Tensor) -> torch.Tensor:
    qmin, qmax= -8.0, 7.0
    max_grad = grad.abs().max()
    if max_grad == 0.0:
        return grad
    scale = max_grad / qmax
    grad_rdown = torch.clamp( (grad / scale), qmin, qmax)
    grad_rup = grad_rdown * scale
    return grad_rup


def fake_quantize(x):
    qmin, qmax= -128.0, 127.0
    max_val = x.abs().max()
    scale = max_val / qmax
    round_down = torch.round(x / scale).clamp(qmin, qmax)
    round_up = round_down * scale
    return round_up

def quant_forward(module, input, output):
    xq = fake_quantize(output)
    if xq.requires_grad:
        xq.register_hook(lambda grad: grad)
    return xq


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-q", "--quantize", action=argparse.BooleanOptionalAction, default=False, help="Enable quantize")
    
    parser.add_argument("-mn", "--model-name", default="Qwen/Qwen3-0.6B-Base", help="pass in model for finetuning")args.per_device_train_batch_size
    parser.add_argument("-train-batch-size", "--per-device-train-batch-size", default=4, help="training batch size")
    parser.add_argument("-eval-batch-size", "--per-device-eval-batch-size", default=4, help="evaluation batch size")
    parser.add_argument("-bn", "--base-model-name", default="Qwen/Qwen3-0.6B-Base", help="pass in base model for quantized evals")
    parser.add_argument("-nmn", "--new-model-name", default=f"Qwen/Qwen3-0.6B-Base-ft", help="pass in model for finetuning")
    parser.add_argument("-qt", "--qtype", default="4bit", help="Type of quantization. Options: [4bit, 8bit, None]")
    parser.add_argument("-wp", "--wb-project", default="CatchAllProject", help="wandb project name")
    parser.add_argument("-wr", "--wb-run-name", default="Placeholder", help="wandb run name")
    parser.add_argument("-lenp", "--max-prompt-length", default=512, help="prompt length for finetuning")
    parser.add_argument("-lenc","--max-completion-length", default=512, help="completion length for finetuning")
    parser.add_argument("-log","--logging-steps", default=5, help="number of steps between each training log step")
    parser.add_argument("-evalstep","--eval-steps", default=10, help="number of steps between each evaluation step")
    parser.add_argument("-chkpt","--resume-from-checkpoint", action=argparse.BooleanOptionalAction, default=False, help="retrieve from checkpoint")
    parser.add_argument("-eval","--evaluate", action=argparse.BooleanOptionalAction, default=False, help="run trainer.evaluate")
    parser.add_argument("-evalnum", "--eval-num-instances-per-split", default=3, help="Number of instances per eval dataset to be chosen")
    parser.add_argument("-train", "--train-dataset", default="./datasets/train/math_12k")
    parser.add_argument("-evaldata", "--eval-dataset", default="./datasets/evaluation_suite")
    parser.add_argument("-verbose", "--logging-verbose", action=argparse.BooleanOptionalAction, default=True, help="Log prompts and completions")
    parser.add_argument("-save", "--save-steps", default=20, help="Number of steps after which to save model checkpoint")
    parser.add_argument("-qm", "--qwen-math", action=argparse.BooleanOptionalAction, default=False, help="Apply qwen math format")
    parser.add_argument("-g", "--gc", action=argparse.BooleanOptionalAction, default=False, help="use gradient checkpointing")
    parser.add_argument("-ud", "--use-dense", action=argparse.BooleanOptionalAction, default=False, help="use dense rewards")
    parser.add_argument("-ad8", "--adam8", action=argparse.BooleanOptionalAction, default=False, help="use adam 8bit")
    parser.add_argument("-grad8", "--quant-gradient", action=argparse.BooleanOptionalAction, default=False, help="use 8bit gradient casting (fake quantization)")
    parser.add_argument("-adpath", "--adapter-path", default="", help="Path to lora adapter")
    parser.add_argument("-ftd", "--ft-done", action=argparse.BooleanOptionalAction, default=False, help="Fine tuning done")
    parser.add_argument("-ptq", "--ptq", action=argparse.BooleanOptionalAction, default=False, help="Apply PTQ")
    parser.add_argument("-ptq-type", "--ptq-type", default="bnb-4bit", help="Type of PTQ. Options currently supported: [bnb-4bit, bnb-8bit, awq-4, awq-8]")
    parser.add_argument("-oom", "--torch-oom", action=argparse.BooleanOptionalAction, default=False, help="Set if you encounter torch cuda out of memory errors. Disables vllm.")
    parser.add_argument("-drgrpo", "--drgrpo", action=argparse.BooleanOptionalAction, default=False, help="Set for drgrpo evals")

    return parser.parse_args()


def main():

    args = parse_arguments()

    if args.logging_verbose:
        logging.set_verbosity_info()

    if args.quantize and args.qtype=="None":
        raise RuntimeError("Cannot support quantization without type specification (specify --qtype!)")

    prompts_data, eval_data = load_data_for_train_and_eval(args)
    
    model_name = args.model_name

    new_model_name = args.new_model_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if args.quantize and args.qtype != "8bit":
        new_model_name= f"{new_model_name}-quant-{args.qtype}"
        quant_config = get_quant_config(args.qtype)
        # Set device map to cpu for very large models.
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda", 
            quantization_config=quant_config,
            trust_remote_code=True,
        )

        base_model.config.use_cache = False
        base_model.gradient_checkpointing_disable()
        if args.ft_done:
            p_model = PeftModel.from_pretrained(base_model,args.adapter_path)
            model = p_model.merge_and_unload()
        else:
            base_model = prepare_model_for_kbit_training(base_model)
            base_model.gradient_checkpointing_disable()  
            loraconf = LoraConfig(
                        r=8,
                        lora_alpha=16, 
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
                        exclude_modules=["lm_head"],
                        lora_dropout=0.05, 
                        bias="none", 
                        task_type="CAUSAL_LM"
                    )
            if args.resume_from_checkpoint:
                model = PeftModel.from_pretrained(
                        base_model,
                        args.adapter_path,
                        is_trainable=True,
                    )
            else:
                model = get_peft_model(base_model, loraconf)
    elif not args.quantize or args.qtype=="8bit":
        # Set device map to cpu for very large models.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # device_map="auto", 
            trust_remote_code=True,
        )

    os.environ["WANDB_PROJECT"]=args.wb_project

    if args.qtype=="8bit":
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_hook(quant_forward)
                
    if args.quant_gradient:
        assert args.qtype != "None" 
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(quant_grad)
    
    training_args = get_train_args(new_model_name, args)

    reward_fn = Reward(tokenizer, args.use_dense, False)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=prompts_data,
        eval_dataset= eval_data,
        processing_class=tokenizer,
        reward_funcs=[reward_fn], 
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} \n ({100 * trainable_params / total_params:.2f}% of total)")

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    print("Training finished.")

    print(f"Saving model to ./{new_model_name}")
    trainer.save_model(f"./{new_model_name}")

    print("Script finished successfully!")


if __name__=="__main__":
    main()
