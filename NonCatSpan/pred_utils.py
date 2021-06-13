import collections
import json
import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def post_processing_function(examples, features, pred_logits, args, model):
    predictions = postprocess_qa_predictions_with_beam_search(
        args=args,
        examples=examples,
        features=features,
        pred_logits=pred_logits,
        n_best=args.n_best,
        max_ans_len=args.max_ans_len,
        start_n_top=model.config.start_n_top,
        end_n_top=model.config.end_n_top,
    )

    references = {ex[args.id_col]: (ex[args.active_col], ex[args.value_col]) for ex in examples}

    return predictions, references


def postprocess_qa_predictions_with_beam_search(
    args,
    examples,
    features,
    pred_logits: Tuple[np.ndarray, np.ndarray],
    n_best: int = 20,
    max_ans_len: int = 30,
    start_n_top: int = 5,
    end_n_top: int = 5,
    is_world_process_zero: bool = True,
):
    assert len(pred_logits) == 5, "`pred_logits` should be a tuple with five elements."
    start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = pred_logits

    assert pred_logits[0].shape[0] == features.shape[0], \
            f"Got {pred_logits[0].shape[0]} predictions and {features.shape[0]} features."

    example_id_to_index = {k: i for i, k in enumerate(examples[args.id_col])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        prelim_predictions = []
        for feature_index in feature_indices:
            start_log_prob = start_top_log_probs[feature_index]
            start_indexes = start_top_index[feature_index]
            end_log_prob = end_top_log_probs[feature_index]
            end_indexes = end_top_index[feature_index]
            feature_null_score = cls_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_index = int(start_indexes[i])
                    j_ = i * end_n_top + j
                    end_index = int(end_indexes[j_])
                    if (start_index >= len(offset_mapping) or end_index >= len(offset_mapping) \
                        or offset_mapping[start_index] is None or offset_mapping[end_index] is None):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_ans_len:
                        continue
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_log_prob[i] + end_log_prob[j_],
                            "start_log_prob": start_log_prob[i],
                            "end_log_prob": end_log_prob[j_]
                        }
                    )

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best]
        utter = example[args.utter_col]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = utter[offsets[0]: offsets[1]]

        if len(predictions) == 0:
            predictions.insert(0, {"text": '', "start_logit": -1e-6, "end_logit": -1e-6, "score": -2e-6})
        
        pred_active = int(min_null_score < 0)
        pred_text = predictions[0]["text"]
        if pred_active == 0:
            pred_text = ''
        if pred_active == 1 and len(pred_text) == 0:
            pred_text = "dontcare"

        all_predictions[example[args.id_col]] = (pred_active, pred_text)

    return all_predictions


def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float32)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]
            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

