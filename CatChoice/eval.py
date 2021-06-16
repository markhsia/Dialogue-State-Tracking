import os
import sys
import json
import argparse
import logging
import math
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils import clip_grad_norm_
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    DataCollatorWithPadding,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from data_utils import *
from pred_utils import *
from metrics import compute_metrics

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--valid_batch_size", type=int, default=128)
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(os.path.join(args.target_dir, "args.json"), 'r') as f:
        train_args = json.load(f)
    for k, v in train_args.items():
        if not hasattr(args, k):
            vars(args)[k] = v

# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
# Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    
# Setup logging, we only want one process per machine to log things on the screen.
# accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    
# Load pretrained model and tokenizer
    config = RobertaConfig.from_pretrained(args.target_dir)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.target_dir)
    config.num_labels = 1
    config.problem_type = "multi_label_classification"
    model = RobertaForSequenceClassification.from_pretrained(args.target_dir, config=config)

# Load and preprocess the datasets
    raw_datasets = load_dataset("json", data_files={"valid": args.valid_file})
    cols = raw_datasets["valid"].column_names
    args.id_col = "id"
    args.dial_id_col = "dial_id"
    args.utter_col = "utterances"
    args.service_desc_col = "service_desc"
    args.slot_desc_col = "slot_desc"
    args.value_col = "value"
    args.label_col = "label"
    args.start_col = "start"
    args.end_col = "end"

    valid_examples = raw_datasets["valid"]
    #valid_examples = valid_examples.select(range(10))
    prepare_pred_features = partial(prepare_pred_features, args=args, tokenizer=tokenizer)
    valid_dataset = valid_examples.map(
        prepare_pred_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )

# Create DataLoaders
    data_collator = default_data_collator
    valid_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, batch_size=args.valid_batch_size)
    
# Prepare everything with our accelerator.
    model, valid_dataloader = accelerator.prepare(
        model, valid_dataloader
    )

# Evaluate!
    logger.info("\n******** Running evaluating ********")
    logger.info(f"Num valid examples = {len(valid_dataset)}")
    valid_dataset.set_format(columns=["attention_mask", "input_ids"])
    model.eval()
    all_logits = []
    for step, data in enumerate(valid_dataloader):
            with torch.no_grad():
                outputs = model(**data)
                all_logits.append(accelerator.gather(outputs.logits).squeeze(-1).cpu().numpy())

    outputs_numpy = np.concatenate(all_logits, axis=0)

    valid_dataset.set_format(columns=list(valid_dataset.features.keys()))
    predictions, references = post_processing_function(valid_examples, valid_dataset, outputs_numpy, args)
    eval_results = compute_metrics(predictions, references, \
                                valid_examples[args.id_col], valid_examples[args.dial_id_col])
    logger.info("Valid | MGA: {:.5f}, JGA: {:.5f}".format( \
                    eval_results["mga"], eval_results["jga"]))
