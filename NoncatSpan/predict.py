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
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizerFast,
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
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--out_file", type=str, default="./results.json")
    parser.add_argument("--n_best", type=int, default=20)
    parser.add_argument("--max_ans_len", type=int, default=30)
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
    config = XLNetConfig.from_pretrained(args.target_dir)
    tokenizer = XLNetTokenizerFast.from_pretrained(args.target_dir)
    config.__dict__["start_n_top"] = args.start_n_top
    config.__dict__["end_n_top"] = args.end_n_top
    model = XLNetForQuestionAnswering.from_pretrained(args.target_dir, config=config)

# Load and preprocess the datasets
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    cols = raw_datasets["test"].column_names
    args.id_col = "id"
    args.dial_id_col = "dial_id"
    args.utter_col = "utterances"
    args.service_col = "service"
    args.service_desc_col = "service_desc"
    args.slot_col = "slot"
    args.slot_desc_col = "slot_desc"
    args.active_col = "active"
    args.value_col = "value"

    test_examples = raw_datasets["test"]
    #test_examples = test_examples.select(range(10))
    prepare_pred_features = partial(prepare_pred_features, args=args, tokenizer=tokenizer)
    test_dataset = test_examples.map(
        prepare_pred_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )

# Create DataLoaders
    data_collator = default_data_collator
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.test_batch_size)
    
# Prepare everything with our accelerator.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

# Evaluate!
    logger.info("\n******** Running predicting ********")
    logger.info(f"Num test examples = {len(test_dataset)}")
    test_dataset.set_format(columns=["attention_mask", "input_ids", "token_type_ids"])
    model.eval()
    all_start_top_log_probs = []
    all_start_top_index = []
    all_end_top_log_probs = []
    all_end_top_index = []
    all_cls_logits = []
    for step, data in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**data)
            start_top_log_probs = outputs.start_top_log_probs
            start_top_index = outputs.start_top_index
            end_top_log_probs = outputs.end_top_log_probs
            end_top_index = outputs.end_top_index
            cls_logits = outputs.cls_logits
            all_start_top_log_probs.append(accelerator.gather(start_top_log_probs).cpu().numpy())
            all_start_top_index.append(accelerator.gather(start_top_index).cpu().numpy())
            all_end_top_log_probs.append(accelerator.gather(end_top_log_probs).cpu().numpy())
            all_end_top_index.append(accelerator.gather(end_top_index).cpu().numpy())
            all_cls_logits.append(accelerator.gather(cls_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_end_top_log_probs])  # Get the max_length of the tensor
    start_top_log_probs_concat = create_and_fill_np_array(all_start_top_log_probs, test_dataset, max_len)
    start_top_index_concat = create_and_fill_np_array(all_start_top_index, test_dataset, max_len)
    end_top_log_probs_concat = create_and_fill_np_array(all_end_top_log_probs, test_dataset, max_len)
    end_top_index_concat = create_and_fill_np_array(all_end_top_index, test_dataset, max_len)
    all_cls_logits = np.concatenate(all_cls_logits, axis=0)
    outputs_numpy = (
        start_top_log_probs_concat,
        start_top_index_concat,
        end_top_log_probs_concat,
        end_top_index_concat,
        all_cls_logits,
    )

    test_dataset.set_format(columns=list(test_dataset.features.keys()))
    predictions, _ = post_processing_function(test_examples, test_dataset, outputs_numpy, 
                                            args, tokenizer, model)
    
    dial_mapping = dict()
    slot_mapping = dict()
    for example in test_examples:
        dial_mapping[example[args.id_col]] = example[args.dial_id_col]
        slot_mapping[example[args.id_col]] = "{}-{}".format(example[args.service_col], example[args.slot_col])

    results = {dial_id: dict() for dial_id in test_examples[args.dial_id_col]}
    for id_, (active, value) in predictions.items():
        if active and len(value) > 0:
            results[dial_mapping[id_]][slot_mapping[id_]] = value
    with open(args.out_file, 'w') as f:
        json.dump(results, f)
