def prepare_train_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    if pad_on_right:
        first_sentences = [sv + ": " + sl for sv, sl in zip(examples[args.service_desc_col], examples[args.slot_desc_col])]
        second_sentences = examples[args.utter_col]
    else:
        first_sentences = examples[args.utter_col]
        second_sentences = [sv + ": " + sl for sv, sl in zip(examples[args.service_desc_col], examples[args.slot_desc_col])]
    
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    special_tokens = tokenized_examples.pop("special_tokens_mask")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["is_impossible"] = []
    tokenized_examples["cls_index"] = []
    tokenized_examples["p_mask"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sep_index = input_ids.index(tokenizer.sep_token_id)
        tokenized_examples["cls_index"].append(cls_index)

        sequence_ids = tokenized_examples["token_type_ids"][i]
        for k, s in enumerate(special_tokens[i]):
            if s:
                sequence_ids[k] = 3
        utter_idx = 1 if pad_on_right else 0

        tokenized_examples["p_mask"].append(
            [0.0 if (not special_tokens[i][k] and s == utter_idx) or k in [cls_index, sep_index] else 1.0 \
                for k, s in enumerate(sequence_ids)]
        )

        sample_index = sample_mapping[i]
        impossible = 1 - examples[args.active_col][sample_index]
        char_start_index = examples[args.start_col][sample_index]
        char_end_index = examples[args.end_col][sample_index]
        value = examples[args.value_col][sample_index]
        if impossible == 1: # non-active
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            tokenized_examples["is_impossible"].append(1.0)
        else:
            token_start_index = 0
            while sequence_ids[token_start_index] != utter_idx:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != utter_idx:
                token_end_index -= 1
            
            if not (offsets[token_start_index][0] <= char_start_index \
                    and offsets[token_end_index][1] >= char_end_index): # non-active
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["is_impossible"].append(1.0)
            elif value == "dontcare":   # dontcare
                tokenized_examples["start_positions"].append(sep_index)
                tokenized_examples["end_positions"].append(sep_index)
                tokenized_examples["is_impossible"].append(0.0)
            else:   # active
                while token_start_index <= token_end_index and offsets[token_start_index][0] <= char_start_index:
                    token_start_index += 1
                token_start_index -= 1
                tokenized_examples["start_positions"].append(token_start_index)
                while offsets[token_end_index][1] >= char_end_index:
                    token_end_index -= 1
                token_end_index += 1
                tokenized_examples["end_positions"].append(token_end_index)
                tokenized_examples["is_impossible"].append(0.0)
                
                #if offsets[token_start_index][0] != char_start_index or offsets[token_end_index][1] != char_end_index:
                #    print("OhOhOh", examples[args.id_col][sample_index], char_start_index, char_end_index, token_start_index, token_end_index, offsets[token_start_index], offsets[token_end_index], examples[args.utter_col][sample_index][offsets[token_start_index][0]: offsets[token_end_index][1]], examples[args.value_col][sample_index])
    
    return tokenized_examples

def prepare_pred_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    if pad_on_right:
        first_sentences = [sv + ": " + sl for sv, sl in zip(examples[args.service_desc_col], examples[args.slot_desc_col])]
        second_sentences = examples[args.utter_col]
    else:
        first_sentences = examples[args.utter_col]
        second_sentences = [sv + ": " + sl for sv, sl in zip(examples[args.service_desc_col], examples[args.slot_desc_col])]
    
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    special_tokens = tokenized_examples.pop("special_tokens_mask")
    tokenized_examples["example_id"] = []
    tokenized_examples["cls_index"] = []
    tokenized_examples["p_mask"] = []
    for i, input_ids in enumerate(tokenized_examples["input_ids"]):
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sep_index = input_ids.index(tokenizer.sep_token_id)
        tokenized_examples["cls_index"].append(cls_index)

        sequence_ids = tokenized_examples["token_type_ids"][i]
        for k, s in enumerate(special_tokens[i]):
            if s:
                sequence_ids[k] = 3
        utter_idx = 1 if pad_on_right else 0

        tokenized_examples["p_mask"].append(
            [0.0 if (not special_tokens[i][k] and s == utter_idx) or k in [cls_index, sep_index] else 1.0
                for k, s in enumerate(sequence_ids)]
        )

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples[args.id_col][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == utter_idx else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])]

    return tokenized_examples


