from collections import defaultdict


def compute_metrics(predictions, references, ids, dial_ids):
    dial_id_mapping = defaultdict(list)
    for id_, dial_id in zip(ids, dial_ids):
        dial_id_mapping[dial_id].append(id_)
    
    metrics = dict()
    active_accs = compute_active_accs(predictions, references, dial_id_mapping)
    metrics["maa"] = active_accs["mean"]
    metrics["jaa"] = active_accs["joint"]
    value_ems = compute_value_ems(predictions, references, dial_id_mapping) # non-actives are excluded
    metrics["mem"] = value_ems["mean"]
    metrics["jem"] = value_ems["joint"]
    goal_accs = compute_goal_accs(predictions, references, dial_id_mapping)
    metrics["mga"] = goal_accs["mean"]
    metrics["jga"] = goal_accs["joint"]

    return metrics

def compute_active_accs(predictions, references, dial_id_mapping):
    total_correct_sum = 0
    joint_correct_sum = 0
    for dial_id, ids in dial_id_mapping.items():
        joint_correct = 1
        for id_ in ids:
            if predictions[id_][0] != references[id_][0]:
                joint_correct = 0
            else:
                total_correct_sum += 1
        joint_correct_sum += joint_correct

    return {"mean": total_correct_sum / len(predictions), "joint": joint_correct_sum / len(dial_id_mapping)}

def compute_value_ems(predictions, references, dial_id_mapping):
    total_correct_sum = 0
    total_sum = 0
    joint_correct_sum = 0
    for dial_id, ids in dial_id_mapping.items():
        joint_correct = 1
        for id_ in ids:
            if references[id_][0] == 0: # only consider active slots in this evaluation
                continue
            if predictions[id_][1].lower() != references[id_][1].lower():
                joint_correct = 0
            else:
                total_correct_sum += 1
            total_sum += 1
        joint_correct_sum += joint_correct

    return {"mean": total_correct_sum / total_sum, "joint": joint_correct_sum / len(dial_id_mapping)}

def compute_goal_accs(predictions, references, dial_id_mapping):
    total_correct_sum = 0
    joint_correct_sum = 0
    for dial_id, ids in dial_id_mapping.items():
        joint_correct = 1
        for id_ in ids:
            if predictions[id_][0] != references[id_][0] \
                or predictions[id_][1].lower() != references[id_][1].lower():
                joint_correct = 0
            else:
                total_correct_sum += 1
        joint_correct_sum += joint_correct

    return {"mean": total_correct_sum / len(predictions), "joint": joint_correct_sum / len(dial_id_mapping)}
