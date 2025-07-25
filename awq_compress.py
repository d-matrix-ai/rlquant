from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
# from compressed_tensors.quantization.quant_args import QuantizationArgs, QuantizationStrategy, QuantizationStrategy

import argparse
from train_model import load_data_for_train_and_eval, parse_arguments

# model_name = 'Qwen/Qwen3-0.6B-Base'
model_name = 'Done/Qwen3-600M-ft-genbatch32-gradacc2'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,)
print("\n===========LOADED MODEL===========\n")
args = parse_arguments()
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 512

train = load_from_disk("./datasets/train/math_12k")
ds = train["train"].select(range(128))
ds = ds.shuffle(42)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            [{"role": "user", "content": example["problem"]}],
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

print(f"length of ds: {len(ds)}")

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        # truncation=True,
        add_special_tokens=False,
    )

print("\n===========LOADED CALIBRATION DATA===========\n")

recipe = [AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),]
# recipe = [AWQModifier(ignore=["lm_head"], scheme="W8A16", targets=["Linear"]),]

# recipe = [AWQModifier(ignore=["lm_head"], scheme="W8A16_ASYM", targets=["Linear"])),]


oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=MAX_SEQUENCE_LENGTH, num_calibration_samples=NUM_CALIBRATION_SAMPLES,)
print("\n===========ONESHOT DONE===========\n")

print("\n\n")

print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt",padding=True).input_ids
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))

print("==========================================\n\n")
new_model_name = f"Qwen3-0.6B-Base-ft-awq-4bit-test-updates"
model.save_pretrained(new_model_name, save_compressed=True)
tokenizer.save_pretrained(new_model_name)