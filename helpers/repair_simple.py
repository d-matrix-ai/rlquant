import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from train_model import Reward, get_quant_config, load_data_for_train_and_eval, parse_arguments
from evaluate import Collate_Fn
from torch.utils.data import DataLoader
from tqdm import tqdm 
import numpy as np
import statistics
import wandb


wandb_project = "Qwen3-1.7-Evals-temp0"
wandb_run = "Qwen3-1.7B-qt-ste8"

base_path = "Qwen/Qwen3-1.7B-Base"
ft_path = "Qwen3-1.7B-ft-8bit-autograd"

def compute_scale(W):
    return torch.clamp(W.abs().max()/127, 1e-6)

def get_submodule(model, name):
    if hasattr(model, "get_submodule"):
        return model.get_submodule(name)

    mod = model
    for attr in name.split("."):
        mod = getattr(mod, attr)
    return mod

args = parse_arguments()
args.train_dataset=""
_, eval_data = load_data_for_train_and_eval(args)

base_model = AutoModelForCausalLM.from_pretrained(base_path, trust_remote_code=False, device_map="cpu", low_cpu_mem_usage=False,)

base_sd = base_model.state_dict()

def get_base_weight(base_sd, name):
    k = f"{name}.weight"
    if k in base_sd: return base_sd[k]
    k2 = f"model.{name}.weight"
    return base_sd.get(k2, None)

def normalize_name_for_base(name: str) -> str:
    if name.startswith("module."):
        name = name[len("module."):]
    return name


tokenizer = AutoTokenizer.from_pretrained(ft_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

from module import QuantizedLinear
from accelerate.state import AcceleratorState
model = AutoModelForCausalLM.from_pretrained(ft_path, trust_remote_code=True, device_map= "cpu")
accelerator=Accelerator()
AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 8

model = accelerator.prepare(model)

for name, module in model.named_modules():
    if isinstance(module, QuantizedLinear):
        base_name = normalize_name_for_base(name)
        weight = get_base_weight(base_sd, base_name)
        if weight is None or weight.numel() == 0:
            print(f"!!!ERROR!!!!{name} is none or has 0 numel")
            exit()
        # max_val = weight.abs().max()
        scale = compute_scale(weight)
        module.scale = scale.to(module.linear.weight.device).to(dtype=module.linear.weight.dtype)

model.eval()

print("!!!!!Model loaded!!!!!")

reward_fn = Reward(tokenizer, False, True)

if accelerator.is_main_process:
        wandb.init(project=wandb_project, name=wandb_run)
    
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