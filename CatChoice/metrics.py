from collections import defaultdict

def compute_metrics(predictions, references, ids, dial_ids):
    dial_id_mapping = defaultdict(set)
    for id_, dial_id in zip(ids, dial_ids):
        dial_id_mapping[dial_id].add(id_)
    
    metrics = dict()
    goal_accs = compute_goal_accs(predictions, references, dial_id_mapping)
    metrics["mga"] = goal_accs["mean"]
    metrics["jga"] = goal_accs["joint"]

    return metrics

def compute_goal_accs(predictions, references, dial_id_mapping):
    total_correct_sum = 0
    joint_correct_sum = 0
    for dial_id, ids in dial_id_mapping.items():
        joint_correct = 1
        for id_ in ids:
            if predictions[id_].lower() not in [v.lower() for v in references[id_]]:
                joint_correct = 0
            else:
                total_correct_sum += 1
        joint_correct_sum += joint_correct

    return {"mean": total_correct_sum / len(predictions), "joint": joint_correct_sum / len(dial_id_mapping)}
