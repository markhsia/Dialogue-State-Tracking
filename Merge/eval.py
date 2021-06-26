import os
import json
import argparse
import glob
from collections import defaultdict
import random

random.seed(1114)

def get_feasible_slots(case, schema):
    feasible_slots = set()
    feasible_slots_dict = defaultdict(set)
    for service_i, service_chunk in enumerate(schema):
        service = service_chunk["service_name"]
        for slot_i, slot_chunk in enumerate(service_chunk["slots"]):
            slot = slot_chunk["name"]
            feasible_cases = set(["all"])
            if not slot_chunk["is_categorical"]:
                feasible_cases.add("noncat")
            else:
                feasible_cases.add("cat")
                if sorted([v.lower() for v in slot_chunk["possible_values"]]) == ["false", "true"]:
                    feasible_cases.add("cat_bool")
                elif all([v.isnumeric() for v in slot_chunk["possible_values"]]):
                    feasible_cases.add("cat_num")
                else:
                    feasible_cases.add("cat_text")
            if case in feasible_cases:
                feasible_slots.add("{}-{}".format(service, slot))
                feasible_slots_dict[service].add(slot)
    
    return feasible_slots, feasible_slots_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dial_dir", required=True, type=str)
    parser.add_argument("-s", "--schema_file", required=True, type=str)
    parser.add_argument("-i", "--in_file", required=True, type=str)
    parser.add_argument("-c", "--case", default="all", type=str)
    args = parser.parse_args()
    
    with open(args.schema_file, 'r') as rf:
        schema = json.load(rf)
    feasible_slots, feasible_slots_dict = get_feasible_slots(args.case, schema)
    noncat_slots, _ = get_feasible_slots("noncat", schema)

    gts = defaultdict(dict)
    for fn in sorted(glob.glob(os.path.join(args.dial_dir, "dialogues_*.json"))):
        with open(fn, 'r') as rf:
            dials = json.load(rf)
            
        for dial in dials:
            dial_id = dial["dialogue_id"]
            services = set(dial["services"])
            utterances = ' '.join([turn["utterance"] for turn in dial["turns"]]).lower()
            states_record = dict()
            for turn in dial["turns"]:
                for frame in turn["frames"]:
                    service = frame["service"]
                    if "state" in frame:
                        states_record[service] = frame["state"]["slot_values"]
            for service in services:
                for slot in feasible_slots_dict[service]:
                    values = states_record[service].get(slot, [])
                    slot = "{}-{}".format(service, slot)
                    if slot not in feasible_slots:
                        continue
                    elif len(values) == 0:
                        gts[dial_id]
                        continue
                    values = [v.lower() for v in values]
                    value = values[0]
                    if slot in noncat_slots:
                        for v in values:
                            if v in utterances or v == "dontcare":
                                value = v
                                break
                    gts[dial_id][slot.lower()] = value
    
    preds = defaultdict(dict)
    with open(args.in_file, 'r') as rf:
        ori_preds = json.load(rf)
    for dial_id, d in ori_preds.items():
        for k, v in d.items():
            if k in feasible_slots:
                preds[dial_id][k.lower()] = v.lower()
    
    joint_correct_sum = 0
    joint_total_sum = 0
    for dial_id, gt in gts.items():
        gt = sorted(gt.items(), key=lambda x: x[0])
        pred = sorted(preds[dial_id].items(), key=lambda x: x[0])
        joint_total_sum += 1
        if gt == pred:
            joint_correct_sum += 1
    print("{} JGA = {:.5f}".format(args.case, joint_correct_sum / joint_total_sum))
