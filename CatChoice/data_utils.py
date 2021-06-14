import numpy as np

def prepare_train_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    if pad_on_right:
        first_sentences = [sv + ": " + sl + ": " + v for sv, sl, v in zip(examples[args.service_col], \
                                                                            examples[args.slot_col], \
                                                                            examples[args.value_col])]
        second_sentences = examples[args.utter_col]
    else:
        first_sentences = examples[args.utter_col]
        second_sentences = [sv + ": " + sl + ": " + v for sv, sl, v in zip(examples[args.service_col],
                                                                            examples[args.slot_col], \
                                                                            examples[args.value_col])]
    
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["labels"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)
        utter_idx = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        label = examples[args.label_col][sample_index]
        char_start_index = examples[args.start_col][sample_index]
        char_end_index = examples[args.end_col][sample_index]
        value = examples[args.value_col][sample_index]
        if label == 0:
            tokenized_examples["labels"].append(0.0)
        elif value == "unknown":
            tokenized_examples["labels"].append(1.0)
        else:
            token_start_index = 0
            while sequence_ids[token_start_index] != utter_idx:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != utter_idx:
                token_end_index -= 1
            
            if not (offsets[token_start_index][0] <= char_start_index \
                    and offsets[token_end_index][1] >= char_end_index):
                tokenized_examples["labels"].append(0.0)
            else:
                tokenized_examples["labels"].append(1.0)
     
    return tokenized_examples

def prepare_pred_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    if pad_on_right:
        first_sentences = [sv + ": " + sl + ": " + v for sv, sl, v in zip(examples[args.service_col], \
                                                                            examples[args.slot_col], \
                                                                            examples[args.value_col])]
        second_sentences = examples[args.utter_col]
    else:
        first_sentences = examples[args.utter_col]
        second_sentences = [sv + ": " + sl + ": " + v for sv, sl, v in zip(examples[args.service_col],
                                                                            examples[args.slot_col], \
                                                                            examples[args.value_col])]

    
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["example_id"] = []
    for i, input_ids in enumerate(tokenized_examples["input_ids"]):
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples[args.id_col][sample_index])

def get_balance_params(labels, neg_ratio):
    labels = np.array(labels)
    pos_sum = np.sum(labels)
    neg_sum = pos_sum * neg_ratio
    pos_w = 1
    neg_w = neg_sum / np.sum(1 - labels)
    weights = np.where(labels == 1, pos_w, neg_w)
    sampling_num = int(pos_sum * (1 + neg_ratio))
    print(np.sum(weights * labels), np.sum(weights * (1 - labels)))
    
    return weights, sampling_num
