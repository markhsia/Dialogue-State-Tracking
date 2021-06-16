import json
import logging
import os
from collections import defaultdict
import numpy as np


logger = logging.getLogger(__name__)

def post_processing_function(examples, features, pred_logits, args):
    predictions = postprocess_choice_predictions(
        args=args,
        examples=examples,
        features=features,
        pred_logits=pred_logits,
    )
    references = postprocess_choice_references(
        args=args,
        examples=examples
    )
    assert len(predictions) == len(references)

    return predictions, references

def postprocess_choice_predictions(
    args,
    examples,
    features,
    pred_logits,
    is_world_process_zero: bool = True,
):
    assert pred_logits.shape[0] == features.shape[0], \
            f"Got {len(pred_logits[0])} pred_logits and {len(features)} features."

    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info("Post-processing example predictions.")
    
    prelim_predictions = defaultdict(list)
    for feature_index, feature in enumerate(features):
        prelim_predictions[feature["example_id"]].append(
            {
                "score": pred_logits[feature_index],
                "value": feature["value"]
            }
        )
    
    all_predictions = {k: sorted(v, key=lambda x: x["score"], reverse=True)[0]["value"] \
                            for k, v in prelim_predictions.items()}
    
    return all_predictions


def postprocess_choice_references(
    args,
    examples,
    is_world_process_zero: bool = True,
):
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info("Post-processing example references.")
    
    all_references = defaultdict(list)
    for example in examples:
        if example[args.label_col] in [1, -1]:
            all_references[example[args.id_col]].append(example[args.value_col])
    
    for k, v in all_references.items():
        assert len(v) > 0

    return all_references
