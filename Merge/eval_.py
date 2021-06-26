import os
import json
import argparse
import glob
from collections import defaultdict
import random

random.seed(1114)

def get_feasible_slots(case, schema):
    feasible_slots = set()
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
                feasible_slots.add("{}-{}".format(service, slot).lower())
    
    return feasible_slots

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dial_dir", required=True, type=str)
    parser.add_argument("-s", "--schema_file", required=True, type=str)
    parser.add_argument("-i", "--in_file", required=True, type=str)
    parser.add_argument("-c", "--case", default="all", type=str)
    args = parser.parse_args()
    
    with open(args.schema_file, 'r') as rf:
        schema = json.load(rf)
    feasible_slots = get_feasible_slots(args.case, schema)
    noncat_slots = get_feasible_slots("noncat", schema)
 
    gts = defaultdict(dict)
    id_set = set()
    #with open("../NoncatSpan/task_data/dev.jsonl", 'r') as f:
    #    for line in f:
    #        d = json.loads(line)
    #        gts[d["dial_id"]]
    #        if d["active"] == 1:
    #            gts[d["dial_id"]]["{}-{}".format(d["service"], d["slot"]).lower()] = d["value"]
    with open("../CatChoice/task_data/dev.jsonl", 'r') as f:
        for line in f:
            d = json.loads(line)
            gts[d["dial_id"]]
            if d["label"] == 1 and d["value"] != "unknown":
                gts[d["dial_id"]]["{}-{}".format(d["service"], d["slot"]).lower()] = d["value"]
    
    preds = dict()
    with open(args.in_file, 'r') as rf:
        ori_preds = json.load(rf)
    for dial_id, d in ori_preds.items():
        preds[dial_id] = dict()
        for k, v in d.items():
            k = k.lower()
            v = v.lower()
            if k in feasible_slots:
                preds[dial_id][k] = v
    #assert len(gts) == len(preds), (len(gts), len(preds))
    joint_correct_sum = 0
    joint_total_sum = 0
    for dial_id, gt in gts.items():
        #if len(gt) == 0:
        #    continue
        gt = sorted(gt.items(), key=lambda x: x[0])
        pred = sorted(preds[dial_id].items(), key=lambda x: x[0])
        #print(dial_id)
        #print(gt)
        #print(pred)
        #print('\n')
        joint_total_sum += 1
        if gt == pred:
            joint_correct_sum += 1
    print(joint_correct_sum, joint_total_sum)
    print("JGA = {:.5f}".format(joint_correct_sum / joint_total_sum))
