from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

import argparse
from train_model import load_data_for_train_and_eval, parse_arguments

def compress_model(args):

    use_train_data = True

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,)
    args = parse_arguments()

    if use_train_data:
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
    else:
        DATASET_SPLIT = "validation"
        DATASET_ID = "Salesforce/wikitext"
        ds = load_dataset(DATASET_ID, "wikitext-2-raw-v1", split=f"{DATASET_SPLIT}[:{128}]")
        ds = ds.shuffle(seed=42)

        def preprocess(example):
            return {
                "text": tokenizer.apply_chat_template(
                    [{"role": "user", "content": example["text"]}],
                    tokenize=False,
                )
            }

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=512,
            add_special_tokens=False,
        )

    if args.ptq_type == "awq-4":
        recipe = [AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),]
    else:
        recipe = [AWQModifier(ignore=["lm_head"], scheme="W8A16", targets=["Linear"]),]
    
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=512, num_calibration_samples=128,)
    print("\n===========ONESHOT DONE===========\n")

    new_model_name = args.model_name + f"-{args.ptq_type}"
    model.save_pretrained(new_model_name, save_compressed=True)
    tokenizer.save_pretrained(new_model_name)
    print("========== COMPRESSED MODEL SAVED ==============")
    return new_model_name
