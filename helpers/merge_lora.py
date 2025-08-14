import os
import argparse 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from train_model import parse_arguments

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

def merge_lora():
    args = parse_arguments
    model_name = args.base_model_name
    base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            # device_map = "cpu"
        )
        base_model.config.use_cache = False
        p_model = PeftModel.from_pretrained(base_model,args.adapter_path).to(device="cuda")
        model = p_model.merge_and_unload()
        new_model_name = model_name + f"-qat-merged"
        model.save_pretrained(new_model_name, save_compressed=True)
        print(f"=============Evaluate model : {new_model_name}=================")
        exit()

if __name__=="__main__":
    merge_lora()
