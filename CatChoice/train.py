import os
import sys
import argparse
import logging
import math
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import WeightedRandomSampler
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
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
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--grad_max_norm", type=int, default=5)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--sched_type", type=str, default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--log_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=32000)
    parser.add_argument("--saved_dir", type=str, default="./saved")
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()
    
    args.saved_dir = os.path.join(args.saved_dir, strftime("%m%d-%H%M", localtime()))
    os.makedirs(args.saved_dir, exist_ok=True)
    
    return args

if __name__ == "__main__":
# Parse arguments and save them.
    args = parse_args()
    logger.info("Saving args to {}...".format(os.path.join(args.saved_dir, "args.json")))
    with open(os.path.join(args.saved_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
# Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.addHandler(logging.FileHandler(os.path.join(args.saved_dir, "log")))
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

# If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
# Load pretrained tokenizer and model. Also, save tokenizer.
    if args.config_name:
        config = RobertaConfig.from_pretrained(args.config_name)
    elif args.model_name:
        config = RobertaConfig.from_pretrained(args.model_name)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    config.num_labels = 1
    config.problem_type = "multi_label_classification"
    
    if args.tokenizer_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_name)
    elif args.model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    logger.info("Saving tokenizer to {}...".format(os.path.join(args.saved_dir, "tokenizer")))
    tokenizer.save_pretrained(args.saved_dir)

    if args.model_name:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name, config=config)
    else:
        logger.info("Training new model from scratch")
        model = RobertaForSequenceClassification.from_config(config)


# Load and preprocess the datasets
    if args.valid_file:
        raw_datasets = load_dataset("json", data_files={"train": args.train_file, "valid": args.valid_file})
    else:
        raw_datasets = load_dataset("json", data_files={"train": args.train_file})
    cols = raw_datasets["train"].column_names
    args.id_col = "id"
    args.dial_id_col = "dial_id"
    args.utter_col = "utterances"
    args.service_desc_col = "service_desc"
    args.slot_desc_col = "slot_desc"
    args.value_col = "value"
    args.label_col = "label"
    args.start_col = "start"
    args.end_col = "end"
    
    train_examples = raw_datasets["train"]
    #train_examples = train_examples.select(range(10))
    prepare_train_features = partial(prepare_train_features, args=args, tokenizer=tokenizer)
    train_dataset = train_examples.map(
        prepare_train_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )
    
    if args.valid_file:
        valid_examples = raw_datasets["valid"]
        #valid_examples = valid_examples.select(range(11))
        prepare_pred_features = partial(prepare_pred_features, args=args, tokenizer=tokenizer)
        valid_dataset = valid_examples.map(
            prepare_pred_features,
            batched=True,
            num_proc=4,
            remove_columns=cols,
        )
    

# Create DataLoaders
    weights = get_balance_weights(train_dataset["labels"], args.neg_ratio)
    sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    data_collator = default_data_collator
    train_dataloader = DataLoader(train_dataset, sampler=sampler, collate_fn=data_collator, 
                                    batch_size=args.train_batch_size, num_workers=4)
    if args.valid_file:
        data_collator = default_data_collator
        valid_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, 
                                    batch_size=args.valid_batch_size, num_workers=4)
    
# Optimizer
# Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_gparams = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_gparams, lr=args.lr)
    
# Prepare everything with our accelerator.
    if args.valid_file:
        model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
    
# Scheduler and math around the number of training steps.
    update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    args.max_update_steps = args.epoch_num * update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.sched_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_update_steps * args.warmup_ratio),
        num_training_steps=args.max_update_steps,
    )

# Train!
    total_train_batch_size = args.train_batch_size * accelerator.num_processes * args.grad_accum_steps
    logger.info("\n******** Running training ********")
    logger.info(f"Num train examples = {len(train_dataset)}")
    logger.info(f"Num Epochs = {args.epoch_num}")
    logger.info(f"Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"Total train batch size (w/ parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"Instantaneous steps per epoch = {len(train_dataloader)}")
    logger.info(f"Update steps per epoch = {update_steps_per_epoch}")
    logger.info(f"Total update steps = {args.max_update_steps}")
    
    max_valid_jga = 0
    for epoch in range(args.epoch_num):
        logger.info("\nEpoch {:02d} / {:02d}".format(epoch + 1, args.epoch_num))
        total_loss = 0
        for step, data in enumerate(train_dataloader, 1):
            model.train()
            outputs = model(**data)
            loss = outputs.loss
            total_loss += loss.item()
            if len(train_dataloader) % args.grad_accum_steps != 0 \
                    and len(train_dataloader) - step < args.grad_accum_steps:
                loss = loss / (len(train_dataloader) % args.grad_accum_steps)
            else:
                loss = loss / args.grad_accum_steps
            accelerator.backward(loss)
            
        # Update model parameters
            if step % args.grad_accum_steps == 0 or step == len(train_dataloader):
                clip_grad_norm_(model.parameters(), max_norm=args.grad_max_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        # Log train loss
            if step % args.log_steps == 0 or step == len(train_dataloader):
                logger.info("Train | Loss: {:.5f}".format(total_loss / step))
        # Evaluate!
            if args.valid_file and (step % args.eval_steps == 0 or step == len(train_dataloader)):
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
                valid_jga = eval_results["jga"]
                if valid_jga >= max_valid_jga:
                    max_valid_jga = valid_jga
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.saved_dir, save_function=accelerator.save)
                    logger.info("Saving config and model to {}...".format(args.saved_dir))
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.saved_dir, save_function=accelerator.save)
    logger.info("Saving config and model to {}...".format(args.saved_dir))
